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

Extract the :file:`tar.gz` file and move the folder to desired destination. 

The first time you install 3D Slicer you should run::

   sudo apt-get install libpulse-dev libnss3 libglu1-mesa 
   sudo apt-get install --reinstall libxcb-xinerama0 

Run the Slicer executable, open the extensions manager (View -> Extension Manager), and install `NvidiaAIAssistedAnnotation <https://github.com/NVIDIA/ai-assisted-annotation-client/blob/master/slicer-plugin/README.md>`_ and `MONAILabel <https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/slicer>`_.

.. _MONAI-Label-plugin-install:

MONAI Label plugin
==================

| GitHub: https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/slicer
| :ref:`Usage instructions <MONAI-Label-plugin-usage>`

The plugin should be installed in 3D Slicer from -> Extension Manager -> Active Learning -> MONAI Label

Once installed, the module will be found in the Modules dropdown under Active Learning -> MONAILabel. Input the server address (http://127.0.0.1:8000/) then pick the app name and volume.
If you get an error that you need to upgrade, you need to install it in developer mode.

Developer mode install
----------------------

Uninstall the non-working MONAI Label extension from 3D Slicer and restart the application.

Go to the location you want to download the MONAI Label source code and run::

    git clone git@github.com:Project-MONAI/MONAILabel.git

In 3D Slicer go to Edit -> Application Settings -> Modules -> Additional Module Paths and add a new module path: :file:`<FULL_PATH>/plugins/slicer/MONAILabel`. You can drag and drop this folder into 3D Slicer instead of typing it out. Restart 3D Slicer.

GIT SSH Key
~~~~~~~~~~~

If you get a message you were denied permission when using :code:`git`, you need to make a SSH key.

Go to your :file:`~/.ssh` folder and generate a key::

    cd ~/.ssh && ssh-keygen

Keep pressing :kbd:`Enter` until you get the files :file:`id_rsa` and :file:`id_rsa.pub`. On GitHub, go to Account settings > SSH and GPG keys

Run :code:`ssh -T git@github.com` for authentication, then return to the location do download the MONAI Label source code and run :code:`git clone git@github.com:Project-MONAI/MONAILabel.git`.
