=================
Creating Datasets
=================

How to make datasets to train models using DICOM imaging and STLs.

The dataset for training should be made up of (the segmentations) and medical imaging correspondng to the labels. These should all be in the compressed Nifti format with extension :file:`nii.gz`.

Exporting from 3D Slicer
========================

Labels
------

The segmentations should be exported from 3D Slicer as labelmaps to DICOM. The :file:`seg.nrrd` file corresponding to the segmentation is a 3D Slicer-specific file, and cannot be direcly converted to a usable format for ML training. If you only have the :file:`seg.nrrd` files, you can still convert them to a binary labelmap without the corresponding medical imaging.

To export a binary labelmap: Right-click the segmentation > "Export visible segments to binary labelmap" (make sure all segments you want to export are visible) > Right-click the new binary labelmap > "Export to DICOM" > Choose the export directory > Export

If you only have STLs, you can import them into 3D Slicer, import the corresponding medical imaging, and create segmentations from the STLs referencing the medical imaging. Then, you can export the segments as a labelmap.

Images
------

The medical imaging should be de-identified for machine learning becuase a dataset without :abbr:`PHI (Patient Health Inforation)` can be more easily shared. Ideally, you should record the original series used in caes you need more specific information such as the exact protocol used. 

Artifacts in the imaging can sometimes be fixed in 3D Slicer if the study is imported through the DICOM Database (as opposed to directly importing the DICOM files to the Data module). If the segments were made with the imaging containing artifacts, the segments may be a little off from the fixed imaging. Make sure you are using the correct imaging with the segments for maximum accuracy.


DICOM Conversions
=================

To make your own medical image datasets, you need to know how to deal with DICOM files.

dcmstack: https://github.com/moloney/dcmstack

dcm2niix
--------

GitHub: https://github.com/rordenlab/dcm2niix
Official page: https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage

`MRIcroGL <https://www.nitrc.org/plugins/mwiki/index.php/mricrogl:MainPage>`_ has a GUI for dcm2niix if you don't want to use the command line.

dcm2nii is obsolete and active development has moved to dcm2niix.

Install: :code:`sudo apt-get install dcm2niix`

Examples:

:code:`dcm2niix -o ~/niftidir -f %p_%s -z y -g y ~/dicomdir`

Convert all DICOM files in current directory to nii.gz:

:code:`dcm2niix -z y ./`

*   Input directory: dicomdir
*   Set output directory: - niftidir
*   Set filenames: -f
*   Generate defaults file: -g y (y stands for "yes")
*   Compress to .gz: -z y. Recommended when dealing with labels since it saves a lot of space.

Convert these one folder at a time to keep things organized.

NRRD Conversions
================

NRRD is a common format for medical images, and is used in 3D Slicer. Your labels may be copied from 3D Slicer as NRRD files.

Converting to nii.gx, taken from https://stackoverflow.com/a/48469229

The following Python code can be used for converting all the '.nrrd' files in a folder into compressed 'nifti' format:

.. code-block:: Python

    import os
    from glob import glob
    import nrrd #pip install pynrrd, if pynrrd is not already installed
    import nibabel as nib #pip install nibabel, if nibabel is not already installed
    import numpy as np

    # You can check your current working directory using os.getcwd() and check the contents using os.listdir().
    # If you are already in the desired directory, the following code can instead be baseDir = os.getcwd()
    baseDir = os.path.normpath('path/to/file')
    files = glob(baseDir+'/*.nrrd')

    for file in files:
        #load nrrd 
        _nrrd = nrrd.read(file)
        data = _nrrd[0]
        header = _nrrd[1]
        #save nifti
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img,os.path.join(baseDir, file[0:-5] + '.nii.gz'))

For example, this script would convert :file:`abc.nrrd` and :file:`xyz.nrrd` files in the :code:`baseDir` to :code:`abc.nii.gz` and :code:`xyz.nii.gz` respectively.


Alt:

.. code-block:: Python

    import os
    import vtk # If you don't have vtk, run: python -m pip install vtk

    def readnrrd(filename):
        """Read image in nrrd format."""
        reader = vtk.vtkNrrdReader()
        reader.SetFileName(filename)
        reader.Update()
        info = reader.GetInformation()
        return reader.GetOutput(), info

    def writenifti(image,filename, info):
        """Write nifti file."""
        writer = vtk.vtkNIFTIImageWriter()
        writer.SetInputData(image)
        writer.SetFileName(filename)
        writer.SetInformation(info)
        writer.Write()

    basepath = os.getcwd()

    m, info = readnrrd(basepath+'test.nrrd')
    writenifti(m, basepath+'mri_prueba2.nii', info)


Inspect 3D Slicer nrrd files:

.. code-block::Python

    pip install slicerio


Make JSON file
==============

Use this code to generate a JSON file for your data:

.. code-block:: bash

    # Assumes all training images are in folder "imagesTr"
    # Assumes all training labels are in folder "labelsTr"
    # Assumes "imagesTr" and "labelsTr" are both in the later defined data_dir directory
    # Assumes corresponding images and labels are named the same
    # User should define the data_dir and edit the text going into the json file
    # About 1/5 of image-label pairs should be for validation
    # This code uses a dataset made of left and right kidney segmentations as an example

    shopt -s nullglob

    root_dir="$MONAI_DATA_DIRECTORY"
    data_dir="${root_dir}/m3D/Kidney volumetrics study/dataset/"
    json_file="${data_dir}dataset.json"

    filelist=()
    for file in "${data_dir}imagesTr"/*.nii.gz; do
       filelist+=("${file##*/}") # Only keep file name
    done

    numTraining=30 # total image-label pairs
    numValidation=6

    echo '{
    "name": "Kidneys",
    "description": "Left and Right Kidney Segmentations",
    "reference": "Ochsner Health",
    "tensorImageSize": "3D",
    "modality": {
    "0": "CT"
    },
    "labels": {
    "0": "background",
    "1": "lkidney",
    "2": "rkidney"
    },
    "numTraining": '"$numTraining"',
    "numTest": '"0"',' > "$json_file"

    training='"training":['
    validation='"validation":['
    pt_1='{"image":"./imagesTr/'
    pt_2='","label":"./labelsTr/'
    pt_3='"},'

    for f in ${filelist[@]:0:$numTraining-$numValidation}; do training="$training$pt_1$f$pt_2$f$pt_3"; done
    for f in ${filelist[@]:$numTraining-$numValidation}; do validation="$validation$pt_1$f$pt_2$f$pt_3"; done

    training="${training::-1}],"
    validation="${validation::-1}]
    }"

    echo "$training" >> "$json_file"
    echo "$validation" >> "$json_file"
