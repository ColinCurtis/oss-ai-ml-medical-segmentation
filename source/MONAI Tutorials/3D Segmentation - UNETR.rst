==================================
Tutorial - 3D Segmentation - UNETR
==================================
Tutorial here: https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unetr_btcv_segmentation_3d.ipynb

Run :code:`python` and set up the environment:

.. code-block:: python

    import os
    import shutil
    import tempfile

    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm

    from monai.losses import DiceCELoss
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        AsDiscrete,
        AddChanneld,
        Compose,
        CropForegroundd,
        LoadImaged,
        Orientationd,
        RandFlipd,
        RandCropByPosNegLabeld,
        RandShiftIntensityd,
        ScaleIntensityRanged,
        Spacingd,
        RandRotate90d,
        ToTensord,
    )

    from monai.config import print_config
    from monai.metrics import DiceMetric
    from monai.networks.nets import UNETR

    from monai.data import (
        DataLoader,
        Dataset,
        CacheDataset,
        load_decathlon_datalist,
        decollate_batch,
    )

    import torch

    print_config()


Set up data directory::

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)


Setup transforms for training and validation:

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

Make an Synapse.org account and install the `Synapse Python Client <https://python-docs.synapse.org/build/html/index.html>`_ using :code:`pip install synapseclient`. This will also install the `Synapse Command Line Client <https://python-docs.synapse.org/build/html/CommandLineClient.html>`_. 

Download dataset from here: https://www.synapse.org/#!Synapse:syn3193805/wiki/. Join the challenge first to access all the files. You can download the files from the command line insted of Python by moving to the desired download directory (preferably :file:`~/MONAI/data/syn3193805`) and running :code:`synapse get -r syn3193805`, however, this may not download the majority of the abdomen and cervix data. If this happens, go into the abdomen and cervix folder individually on Synapse and either download files through the browser or run the code given in :guilabel:`Download Options` > "Programmatic Options" > "Command Line." However, :file:`Abdomen.zip` and :file:`Cervix.zip` duplicate the other files in their respective folders, so don't download those two files. The total size of all the files is about 67 GB.

The :file:`Abdomen` folder contains a folder called :file:`RawData`, which holds the original CT scans and corresponding images. The other two folders are for training and testing with the CT scans after they have already been transformed, which artificially increases the amount of data the model can train with.

The folders you will need for this tutorial are in :file:`Abdomen/RawData`. Move :file:`Testing/img` up to this folder and rename it :file:`imagesTs`. Rename :file:`Training/img` to :file:`imagesTr` and :file:`Training/label` to :file:`labelsTr` and move them up to the :file:`RawData` folder.

Create a JSON file named :file:`dataset_0.json` and paste in this text:

.. raw:: html

   <details>
   <summary><a>dataset_0.json</a></summary>

.. code-block:: JSON

    {
    "description": "btcv yucheng",
    "labels": {
        "0": "background",
        "1": "spleen",
        "2": "rkid",
        "3": "lkid",
        "4": "gall",
        "5": "eso",
        "6": "liver",
        "7": "sto",
        "8": "aorta",
        "9": "IVC",
        "10": "veins",
        "11": "pancreas",
        "12": "rad",
        "13": "lad"
    },
    "licence": "yt",
    "modality": {
        "0": "CT"
    },
    "name": "btcv",
    "numTest": 20,
    "numTraining": 80,
    "reference": "Vanderbilt University",
    "release": "1.0 06/08/2015",
    "tensorImageSize": "3D",
    "test": [
        "imagesTs/img0061.nii.gz",
        "imagesTs/img0062.nii.gz",
        "imagesTs/img0063.nii.gz",
        "imagesTs/img0064.nii.gz",
        "imagesTs/img0065.nii.gz",
        "imagesTs/img0066.nii.gz",
        "imagesTs/img0067.nii.gz",
        "imagesTs/img0068.nii.gz",
        "imagesTs/img0069.nii.gz",
        "imagesTs/img0070.nii.gz",
        "imagesTs/img0071.nii.gz",
        "imagesTs/img0072.nii.gz",
        "imagesTs/img0073.nii.gz",
        "imagesTs/img0074.nii.gz",
        "imagesTs/img0075.nii.gz",
        "imagesTs/img0076.nii.gz",
        "imagesTs/img0077.nii.gz",
        "imagesTs/img0078.nii.gz",
        "imagesTs/img0079.nii.gz",
        "imagesTs/img0080.nii.gz"
    ],
    "training": [
        {
            "image": "imagesTr/img0001.nii.gz",
            "label": "labelsTr/label0001.nii.gz"
        },
        {
            "image": "imagesTr/img0002.nii.gz",
            "label": "labelsTr/label0002.nii.gz"
        },
        {
            "image": "imagesTr/img0003.nii.gz",
            "label": "labelsTr/label0003.nii.gz"
        },
        {
            "image": "imagesTr/img0004.nii.gz",
            "label": "labelsTr/label0004.nii.gz"
        },
        {
            "image": "imagesTr/img0005.nii.gz",
            "label": "labelsTr/label0005.nii.gz"
        },
        {
            "image": "imagesTr/img0006.nii.gz",
            "label": "labelsTr/label0006.nii.gz"
        },
        {
            "image": "imagesTr/img0007.nii.gz",
            "label": "labelsTr/label0007.nii.gz"
        },
        {
            "image": "imagesTr/img0008.nii.gz",
            "label": "labelsTr/label0008.nii.gz"
        },
        {
            "image": "imagesTr/img0009.nii.gz",
            "label": "labelsTr/label0009.nii.gz"
        },
        {
            "image": "imagesTr/img0010.nii.gz",
            "label": "labelsTr/label0010.nii.gz"
        },
        {
            "image": "imagesTr/img0021.nii.gz",
            "label": "labelsTr/label0021.nii.gz"
        },
        {
            "image": "imagesTr/img0022.nii.gz",
            "label": "labelsTr/label0022.nii.gz"
        },
        {
            "image": "imagesTr/img0023.nii.gz",
            "label": "labelsTr/label0023.nii.gz"
        },
        {
            "image": "imagesTr/img0024.nii.gz",
            "label": "labelsTr/label0024.nii.gz"
        },
        {
            "image": "imagesTr/img0025.nii.gz",
            "label": "labelsTr/label0025.nii.gz"
        },
        {
            "image": "imagesTr/img0026.nii.gz",
            "label": "labelsTr/label0026.nii.gz"
        },
        {
            "image": "imagesTr/img0027.nii.gz",
            "label": "labelsTr/label0027.nii.gz"
        },
        {
            "image": "imagesTr/img0028.nii.gz",
            "label": "labelsTr/label0028.nii.gz"
        },
        {
            "image": "imagesTr/img0029.nii.gz",
            "label": "labelsTr/label0029.nii.gz"
        },
        {
            "image": "imagesTr/img0030.nii.gz",
            "label": "labelsTr/label0030.nii.gz"
        },
        {
            "image": "imagesTr/img0031.nii.gz",
            "label": "labelsTr/label0031.nii.gz"
        },
        {
            "image": "imagesTr/img0032.nii.gz",
            "label": "labelsTr/label0032.nii.gz"
        },
        {
            "image": "imagesTr/img0033.nii.gz",
            "label": "labelsTr/label0033.nii.gz"
        },
        {
            "image": "imagesTr/img0034.nii.gz",
            "label": "labelsTr/label0034.nii.gz"
        }
    ],
    "validation": [
        {
            "image": "imagesTr/img0035.nii.gz",
            "label": "labelsTr/label0035.nii.gz"
        },
        {
            "image": "imagesTr/img0036.nii.gz",
            "label": "labelsTr/label0036.nii.gz"
        },
        {
            "image": "imagesTr/img0037.nii.gz",
            "label": "labelsTr/label0037.nii.gz"
        },
        {
            "image": "imagesTr/img0038.nii.gz",
            "label": "labelsTr/label0038.nii.gz"
        },
        {
            "image": "imagesTr/img0039.nii.gz",
            "label": "labelsTr/label0039.nii.gz"
        },
        {
            "image": "imagesTr/img0040.nii.gz",
            "label": "labelsTr/label0040.nii.gz"
        }
    ]

    }

.. raw:: html

   </details>

|
| Define CacheDataset and DataLoader for training and validation, making sure :code:`root_dir + data_dir` points to the contents of the :file:`RawData` folder:

.. code-block:: python

    data_dir = "/syn3193805/Abdomen/RawData/"
    split_JSON = "dataset_0.json"
    datasets = root_dir + data_dir + split_JSON
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = CacheDataset(
        data=datalist,
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

Alternative for troubleshooting memory:

.. code-block:: python

    data_dir = "/syn3193805/Abdomen/RawData/"
    split_JSON = "dataset_0.json"
    datasets = root_dir + data_dir + split_JSON
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = monai.data.Dataset(data=datalist, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=False
    )
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=False
    )

Check data shape and visualize:

.. code-block:: python

    slice_map = {
        "img0035.nii.gz": 170,
        "img0036.nii.gz": 230,
        "img0037.nii.gz": 204,
        "img0038.nii.gz": 204,
        "img0039.nii.gz": 204,
        "img0040.nii.gz": 180,
    }
    case_num = 0
    img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]
    img_shape = img.shape
    label_shape = label.shape
    print(f"image shape: {img_shape}, label shape: {label_shape}")
    plt.figure("image", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(img[0, :, :, slice_map[img_name]].detach().cpu(), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[0, :, :, slice_map[img_name]].detach().cpu())
    plt.show()


Create Model, Loss, Optimizer:

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
                        model.state_dict(), os.path.join(root_dir, "UNETR_Abd_best_metric_model.pth")
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
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )
    model.load_state_dict(torch.load(os.path.join(root_dir, "UNETR_Abd_best_metric_model.pth")))

