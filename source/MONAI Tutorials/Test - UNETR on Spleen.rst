======================
Test - UNETR on Spleen
======================

Differences between UNet tutorial on spleens and UNETR tutorial on multiple anatomical parts.

Run :code:`python`.

.. raw:: html

   <details>
   <summary><a>Code for Imports through loading the JSON data</a></summary>

.. code-block:: python

    import os
    import shutil
    import tempfile
    import matplotlib.pyplot as plt
    import torch
    from monai.transforms import (
        AsDiscrete,
        Compose,
        CropForegroundd,
        LoadImaged,
        Orientationd,
        RandCropByPosNegLabeld,
        ScaleIntensityRanged,
        Spacingd,
    )
    from monai.metrics import DiceMetric
    from monai.inferers import sliding_window_inference
    from monai.data import (
        DataLoader,
        Dataset,
        CacheDataset,
        decollate_batch,
    )
    from monai.config import print_config
    from monai.data import load_decathlon_datalist # Used for loading JSON file
    import glob
    import time
    from monai.utils import first, set_determinism
    from monai.transforms import (
        AsDiscreted,
        EnsureChannelFirstd,
        EnsureTyped,
        EnsureType,
        Invertd,
    )
    from monai.handlers.utils import from_engine
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm
    from monai.losses import DiceLoss
    from monai.apps import download_and_extract
    import numpy as np
    from tqdm import tqdm
    from monai.losses import DiceCELoss
    from monai.transforms import (
        AddChanneld,
        RandFlipd,
        RandShiftIntensityd,
        RandRotate90d,
        ToTensord,
    )
    from monai.networks.nets import UNETR

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)
    # The following would change depending on the data you are working with
    root_dir = os.path.join(root_dir, "MSD")
    data_dir = os.path.join(root_dir, "Task09_Spleen")

    split_JSON = "dataset.json"
    # datasets = data_dir + split_JSON
    datasets = os.path.join(data_dir, split_JSON)
    train_files = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")

    set_determinism(seed=0)

.. raw:: html

   </details>

Imports
=======

Common Imports
--------------
.. code-block:: python

    import os
    import shutil
    import tempfile
    import matplotlib.pyplot as plt
    import torch
    from monai.transforms import (
        AsDiscrete,
        Compose,
        CropForegroundd,
        LoadImaged,
        Orientationd,
        RandCropByPosNegLabeld,
        ScaleIntensityRanged,
        Spacingd,
    )
    from monai.metrics import DiceMetric
    from monai.inferers import sliding_window_inference
    from monai.data import (
        DataLoader,
        Dataset,
        CacheDataset,
        decollate_batch,
    )
    from monai.config import print_config
    from monai.data import load_decathlon_datalist # Used for loading JSON file

UNet unique imports
-------------------
.. code-block:: python
    
    import glob
    import time
    from monai.utils import first, set_determinism
    from monai.transforms import (
        AsDiscreted,
        EnsureChannelFirstd,
        EnsureTyped,
        EnsureType,
        Invertd,
    )
    from monai.handlers.utils import from_engine
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm
    from monai.losses import DiceLoss
    from monai.apps import download_and_extract


UNETR unique imports
--------------------

.. code-block:: python

    import numpy as np
    from tqdm import tqdm
    from monai.losses import DiceCELoss
    from monai.transforms import (
        AddChanneld,
        RandFlipd,
        RandShiftIntensityd,
        RandRotate90d,
        ToTensord,
    )
    from monai.networks.nets import UNETR

Set data path
=============

.. code-block:: python

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)
    # The following would change depending on the data you are working with
    root_dir = os.path.join(root_dir, "MSD")
    data_dir = os.path.join(root_dir, "Task09_Spleen")

Load data
=========

UNet tutorial did not use a JSON file, while UNETR did. If you use the JSON file that came with the spleen dataset, edit it to add about 8 training image/label pairs to a new category "validation."

No JSON file
------------

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


JSON file
---------

.. code-block:: python

    split_JSON = "dataset.json"
    # datasets = data_dir + split_JSON
    datasets = os.path.join(data_dir, split_JSON)
    train_files = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")

Transforms
==========

The UNet tutorial sets deterministic training for reproducibility. Optional for both models::

    set_determinism(seed=0)

The two tutorials have different transforms. They seem to be interchangeable between models.

UNet
----

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

UNETR
-----

.. code-block:: python

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
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
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )

Define CacheDataset and DataLoader
==================================

This is used for training and validation, and the settings here will likely need to be changed if you get errors in training.

UNet
----

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

UNETR
-----

.. code-block:: python

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True
    )
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

Create Model, Loss, Optimizer
=============================

The UNETR model in this step is causing "CUDA out of memory" errors.

UNet
----

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

UNETR
-----

.. code-block:: python

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNETR(
        in_channels=1,
        out_channels=14,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

Execute a typical PyTorch training process
==========================================

UNet
----

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
                        root_dir, "UNet_Task09_Spleen_best_metric_model.pth"))
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

UNETR
-----

.. code-block:: python

    def validation(epoch_iterator_val):
        model.eval()
        dice_vals = list()
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice = dice_metric.aggregate().item()
                dice_vals.append(dice)
                epoch_iterator_val.set_description(
                    "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
                )
            dice_metric.reset()
        mean_dice_val = np.mean(dice_vals)
        return mean_dice_val


    def train(global_step, train_loader, dice_val_best, global_step_best):
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            logit_map = model(x)
            loss = loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
            )
            if (
                global_step % eval_num == 0 and global_step != 0
            ) or global_step == max_iterations:
                epoch_iterator_val = tqdm(
                    val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
                )
                dice_val = validation(epoch_iterator_val)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(
                        model.state_dict(), os.path.join(root_dir, "UNETR_Task09_spleen_best_metric_model.pth")
                    )
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
            global_step += 1
        return global_step, dice_val_best, global_step_best


    max_iterations = 25000
    eval_num = 500
    post_label = AsDiscrete(to_onehot=14)
    post_pred = AsDiscrete(argmax=True, to_onehot=14)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    total_start = time.time()
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )

    total_time = time.time() - total_start
    model.load_state_dict(torch.load(os.path.join(root_dir, "UNETR_Task09_spleen_best_metric_model.pth")))
    print(f"Total time: {total_time}")
