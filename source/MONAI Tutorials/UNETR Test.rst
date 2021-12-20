==========
UNETR Test
==========

Run :code:`python` and run this:

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