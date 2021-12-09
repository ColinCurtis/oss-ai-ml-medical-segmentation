===============
3D Segmentation
===============

UNETR
=====
Tutorial here: https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unetr_btcv_segmentation_3d.ipynb

Make a :file:`data` folder in your :file:`MONAI` folder. Add it as an environment variable to :file:`bashrc` by running :code:`nano ~/.bashrc` and adding :code:`export MONAI_DATA_DIRECTORY=~/MONAI/data` to the end. Apply the change using :code:`source ~/.bashrc`. This allows you to save results and reuse downloads.

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

Download dataset from here: https://www.synapse.org/#!Synapse:syn3193805/wiki/89480\n. Join the challenge first to access all the files. You can download the files from the command line insted of Python by moving to the desired download directory (preferably :file:`~/MONAI/data/syn3193805`) and running :code:`synapse get -r syn3193805`. The total size of the files is about 80 GB. Relabel :file:`averaged-training-images` to :file:`imagesTr` and :file:`averaged-training-labels` to :file:`labelsTr`.