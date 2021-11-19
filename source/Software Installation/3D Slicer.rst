=========
3D Slicer
=========
| Homepage: https://www.slicer.org/ 
| Download: https://download.slicer.org/ 

AIAA requires a preview or stable version of 3D Slicer 4.11.x or newer. 

.. (is this true?)

Uninstall 3D Slicer on Linux by deleting its directory.

Install 3D Slicer
=================

Official installation guide: https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html

Download the Linux Preview Release: https://download.slicer.org/ 

Extract the :file:`tar.gz` file and move the folder to desired destination. Then run::

   sudo apt-get install libpulse-dev libnss3 libglu1-mesa 
   sudo apt-get install --reinstall libxcb-xinerama0 

Run the Slicer executable, open the extensions manager, and install NvidiaAIAssistedAnnotation and MONAILabel.