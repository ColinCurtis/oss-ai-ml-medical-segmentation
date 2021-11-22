=====
MONAI
=====

Medical Open Network for AI

| Homepage: https://monai.io/ 
| Source code: https://github.com/Project-MONAI/MONAI 
| Tutorials: https://github.com/Project-MONAI/tutorials 
| https://www.youtube.com/c/Project-MONAI
| https://docs.monai.io/en/latest/ 
| https://pypi.org/project/monai/ 

New milestone release changes: https://docs.monai.io/en/latest/whatsnew.html

Transfer learning from Clara Train models: https://github.com/Project-MONAI/tutorials/blob/master/modules/transfer_mmar.ipynb

Requires PyTorch 

MONAI Installation
==================

Installation guide: https://docs.monai.io/en/latest/installation.html 

Install current `milestone release <https://pypi.org/project/monai/>`_::
    
   pip install monai 

.. Note::
   Alternatively, you can install the `weekly release <https://pypi.org/project/monai-weekly/>`_ for the latest features::

      pip install monai-weekly

Validate installation by running this, which should return the installed MONAI version information::
    
   python -c 'import monai; monai.config.print_config()' 

Install all optional dependencies::
    
   pip install 'monai[all]' 

MONAI Label
===========

| Homepage: https://docs.monai.io/projects/label/en/latest/
| Source code: https://github.com/Project-MONAI/MONAILabel
| More info: https://medium.com/pytorch/monai-v0-6-and-monai-label-v0-1-e738556b0a10
| Tutorials: https://github.com/Project-MONAI/tutorials

“For Researchers, MONAI Label gives you an easy way to define their pipeline to facilitate the image annotation process.For Clinicians, MONAI Label gives you access to a continuously learning AI that will better understand what the end-user is trying to annotate. 
MONAI Label comprises the following key components: MONAI Label Server, MONAI Label Sample Apps, MONAI Label Sample Datasets, and a 3DSlicer Viewer extension.” 

“Create MONAI Label Apps using the three different paradigms: `DeepGrow <https://github.com/Project-MONAI/MONAILabel/wiki/DeepGrow>`_, `DeepEdit <https://github.com/Project-MONAI/MONAILabel/wiki/DeepEdit>`_, and `automatic segmentation <https://github.com/Project-MONAI/MONAILabel/wiki/Automated-Segmentation>`_.” 

Claims it supports Ubuntu and Windows with GPU/CUDA enabled. 

MONAI Label Installation
------------------------
Official installation guide: https://docs.monai.io/projects/label/en/latest/installation.html 

Install Python libraries::
    
   pip install testresources
   python -m pip install --upgrade pip setuptools wheel 

Install `current milestone release <https://pypi.org/project/monailabel/>`_::
    
   pip install monailabel 

.. Note::
   Alternatively, you can install `the weekly release <https://pypi.org/project/monailabel-weekly/>`_ for the latest features::

      pip install monailabel-weekly

Install from GitHub
~~~~~~~~~~~~~~~~~~~

You can alternatively install the latest version from the GitHub main branch, although it takes longer.

Install git::

   sudo apt install git 

Install the latest version from the GitHub main branch::
    
   pip install git+https://github.com/Project-MONAI/MONAILabel#egg=monailabel 

.. tip::
   If you want to update to the weekly or GitHub version, uninstall monailabel first::

      sudo pip uninstall monailabel

   If that didn't uninstall it, remove :code:`sudo`::

      pip uninstall monailabel

Run :code:`monailabel --help` to see if Monai Label installed correctly. If it says "python: command not found," you may have a :file:`monailabel` file in :file:`~/.local/bin/` that you need to edit. Comment everything out (put a :code:`#` at the beginning of a line) until this is the only executed code::

   set -e
   PYEXE=python3
   ${PYEXE} -m monailabel.main $*
