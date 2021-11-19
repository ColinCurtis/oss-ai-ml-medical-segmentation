=====
MONAI
=====

| Homepage: https://monai.io/ 
| Source code: https://github.com/Project-MONAI/MONAI 
| Tutorials: https://github.com/Project-MONAI/tutorials 
| https://www.youtube.com/c/Project-MONAI
| https://docs.monai.io/en/latest/ 
| https://pypi.org/project/monai/ 

Transfer learning from Clara Train models: https://github.com/Project-MONAI/tutorials/blob/master/modules/transfer_mmar.ipynb

Requires PyTorch 

MONAI Installation
==================

Installation guide: https://docs.monai.io/en/latest/installation.html 

Install current milestone release::
    
   pip install monai 

Validate installation::
    
   python -c 'import monai; monai.config.print_config()' 

Install all optional dependencies::
    
   pip install 'monai[all]' 

MONAI Label
===========

| Homepage: https://docs.monai.io/projects/label/en/latest/
| Source code: https://github.com/Project-MONAI/MONAILabel
| More info: https://medium.com/pytorch/monai-v0-6-and-monai-label-v0-1-e738556b0a10

“For Researchers, MONAI Label gives you an easy way to define their pipeline to facilitate the image annotation process.For Clinicians, MONAI Label gives you access to a continuously learning AI that will better understand what the end-user is trying to annotate. 
MONAI Label comprises the following key components: MONAI Label Server, MONAI Label Sample Apps, MONAI Label Sample Datasets, and a 3DSlicer Viewer extension.” 

“Create MONAI Label Apps using the three different paradigms: `DeepGrow <https://github.com/Project-MONAI/MONAILabel/wiki/DeepGrow>`_, `DeepEdit <https://github.com/Project-MONAI/MONAILabel/wiki/DeepEdit>`_, and `automatic segmentation <https://github.com/Project-MONAI/MONAILabel/wiki/Automated-Segmentation>`_.” 

Claims it supports Ubuntu and Windows with GPU/CUDA enabled. 

MONAI Label Installation
------------------------
Official installation guide: https://docs.monai.io/projects/label/en/latest/installation.html 

Install current milestone release::
    
   pip install monailabel 

Possibly need to run this::
    
   python -m pip install --upgrade pip setuptools wheel 

.. tip::
    If you want to update to a weekly version, uninstall monailabel first::

       sudo pip uninstall monailabel

Install git::

   sudo apt install git 

Install the latest version from the GitHub main branch::
    
   pip install git+https://github.com/Project-MONAI/MONAILabel#egg=monailabel 

As of 2021-09-10, I cannot get :code:`monailabel` or :code:`monai` commands to work in the terminal command line 