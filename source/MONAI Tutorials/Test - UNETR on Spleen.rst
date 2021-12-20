======================
Test - UNETR on Spleen
======================

Differences between UNet tutorial on spleens and UNETR tutorial on multiple anatomical parts

Imports
=======

Common Imports
--------------

UNet
----
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

UNETR
-----

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




.. code-block:: python

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)
    root_dir = os.path.join(root_dir, "MSD")
    data_dir = os.path.join(root_dir, "Task09_Spleen")