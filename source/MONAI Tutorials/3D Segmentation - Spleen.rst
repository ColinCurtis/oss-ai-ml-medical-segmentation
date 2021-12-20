===================================
Tutorial - 3D Segmentation - Spleen
===================================
https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d.ipynb

You should keep all your datasets organized so you don't end up downloading the same datasets again. The following tutorials rely on the environment variable :code:`MONAI_DATA_DIRECTORY` to find the path of your datasets.
Make a :file:`data` folder in your :file:`MONAI` folder. Add it as an environment variable to :file:`bashrc` by running :code:`nano ~/.bashrc` and adding :code:`export MONAI_DATA_DIRECTORY=~/MONAI/data` to the end. Apply the change using :code:`source ~/.bashrc`. This allows you to save results and reuse downloads.

Run :code:`python` and setup environment:

.. code-block:: python

    from monai.utils import first, set_determinism
    from monai.transforms import (
        AsDiscrete,
        AsDiscreted,
        EnsureChannelFirstd,
        Compose,
        CropForegroundd,
        LoadImaged,
        Orientationd,
        RandCropByPosNegLabeld,
        ScaleIntensityRanged,
        Spacingd,
        EnsureTyped,
        EnsureType,
        Invertd,
    )
    from monai.handlers.utils import from_engine
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm
    from monai.metrics import DiceMetric
    from monai.losses import DiceLoss
    from monai.inferers import sliding_window_inference
    from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
    from monai.config import print_config
    from monai.apps import download_and_extract
    import torch
    import matplotlib.pyplot as plt
    import tempfile
    import time
    import shutil
    import os
    import glob

Specify directory with medical data.

.. code-block:: python

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)

If you already have the datasets from the Medical Segmentation Decathalon, you may want to add them to a MSD folder inside a more general datasets folder. If you created a MSD folder, add it to :code:`root_dir`::
    
    root_dir = os.path.join(root_dir, "MSD")

Run this code, which will only download the spleen dataset if you don't already have it:

.. code-block:: python

    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

    compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
    data_dir = os.path.join(root_dir, "Task09_Spleen")
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)

Set MSD Spleen dataset path:

.. code-block:: python

    train_images = sorted(
        glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]


Set deterministic training for reproducibility::

    set_determinism(seed=0)


Setup transforms for training and validation. Here we use several transforms to augment the dataset:

#.   :code:`LoadImaged` loads the spleen CT images and labels from NIfTI format files.
#.   :code:`AddChanneld` as the original data doesn't have channel dim, add 1 dim to construct "channel first" shape.
#.   :code:`Spacingd` adjusts the spacing by :code:`pixdim=(1.5, 1.5, 2.)` based on the affine matrix.
#.   :code:`Orientationd` unifies the data orientation based on the affine matrix.
#.   :code:`ScaleIntensityRanged` extracts intensity range [-57, 164] and scales to [0, 1].
#.   :code:`CropForegroundd` removes all zero borders to focus on the valid body area of the images and labels.
#.   :code:`RandCropByPosNegLabeld` randomly crop patch samples from big image based on pos / neg ratio.
#.   The image centers of negative samples must be in valid body area.
#.   :code:`RandAffined` efficiently performs :code:`rotate`, :code:`scale`, :code:`shear`, :code:`translate`, etc. together based on PyTorch affine transform.
#.   :code:`EnsureTyped` converts the numpy array to PyTorch Tensor for further steps.

.. code-block:: python

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # user can also add other random transforms
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=(96, 96, 96),
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

Check transforms in DataLoader:

.. code-block:: python

    check_ds = Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image, label = (check_data["image"][0][0], check_data["label"][0][0])
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    # plot the slice [:, :, 80]
    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(image[:, :, 80], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[:, :, 80])
    plt.show()


Define CacheDataset and DataLoader for training and validation

Here we use CacheDataset to accelerate training and validation process, it's 10x faster than the regular Dataset.
To achieve best performance, set :code:`cache_rate=1.0` to cache all the data, if memory is not enough, set lower value.
Users can also set :code:`cache_num` instead of :code:`cache_rate`, will use the minimum value of the 2 settings.
And set :code:`num_workers` to enable multi-threads during caching: 4 is usually faster than 0, but 0 will avoid some potential errors.
If want to to try the regular Dataset, just change to use the commented code below.

.. code-block:: python

    num_workers=0
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0, num_workers=num_workers)
    # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=num_workers)

    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=num_workers)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)


Create Model, Loss, Optimizer

.. code-block:: python

    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")


Execute a typical PyTorch training process (this may take a few hours depending on the number of epochs):

.. code-block:: python

    max_epochs = 600
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

    total_start = time.time()
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")\

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)\

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()\

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        # the pth file is renamed from the tutorial
                        # so it doesn't collide with other pth files
                        root_dir, "Task09_Spleen_best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

    total_time = time.time() - total_start
    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")
    print(f"Total time: {total_time}")

.. note::
    If you get the error :code:`RuntimeError: received 0 items of ancdata`, go back to the :code:`CacheDataset` and set :code:`num_workers` to 0. This will take longer but it should work. [Do all of these need to be set to 0? How much longer will this take? Can they be set to another number besides 4 (error) and 0 (takes long)?]

    Try this for the ideal number: http://www.feeny.org/finding-the-ideal-num_workers-for-pytorch-dataloaders/


Plot the loss and metric:

.. code-block:: python

    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.show()


Check best model output with the input image and label:

.. code-block:: python

    model.load_state_dict(torch.load(
        os.path.join(root_dir, "Task09_Spleen_best_metric_model.pth")))
    model.eval()
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(
                val_data["image"].to(device), roi_size, sw_batch_size, model
            )
            # plot the slice [:, :, 80]
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(val_data["label"][0, 0, :, :, 80])
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plt.imshow(torch.argmax(
                val_outputs, dim=1).detach().cpu()[0, :, :, 80])
            plt.show()
            if i == 2:
                break


Evaluation on original image spacings:

.. code-block:: python
        
    val_org_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image"], pixdim=(
                1.5, 1.5, 2.0), mode="bilinear"),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_org_ds = Dataset(
        data=val_files, transform=val_org_transforms)
    val_org_loader = DataLoader(val_org_ds, batch_size=1, num_workers=4)

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=val_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        AsDiscreted(keys="label", to_onehot=2),
    ])

    model.load_state_dict(torch.load(
        os.path.join(root_dir, "Task09_Spleen_best_metric_model.pth")))
    model.eval()

    with torch.no_grad():
        for val_data in val_org_loader:
            val_inputs = val_data["image"].to(device)
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            val_data["pred"] = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)\

        # aggregate the final mean dice result
        metric_org = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

    print("Metric on original image spacing: ", metric_org)


Remove directory if a temporary one was used::

    if directory is None:
        shutil.rmtree(root_dir)

