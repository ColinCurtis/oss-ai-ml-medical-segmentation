===========
MONAI Label
===========

These steps follow this guide: https://docs.monai.io/projects/label/en/latest/installation.html#downloading-sample-apps-or-datasets

Download Sample Apps or Datasets
================================

MONAI Label's location should be in :file:`~/.local/monailabel/`. Here, there should be a :file:`sample-apps` folder with different apps MONAI Label can use to segment. These apps are `explained here <https://github.com/Project-MONAI/MONAILabel/tree/main/sample-apps>`_ and some more `experimental apps are here <https://github.com/diazandr3s/MONAILabel-Apps>`_. List these apps and add them individually to the working directory using this::
   
   monailabel apps # List sample apps
   monailabel apps --download --name deepedit --output apps # Using deepedit as an example

.. Note::
   If :code:`monailabel` install path is not automatically determined, then you can provide explicit install path by adding :code:`--prefix ~/.local`::
   
      monailabel apps --prefix ~/.local
      monailabel apps --prefix ~/.local --download --name deepedit --output apps

Download :abbr:`MSD (Medical Segmentation Decathlon)` `Datasets <https://registry.opendata.aws/msd/>`_ to local directory (`MSD <http://medicaldecathlon.com/>`_ datasets also available `here <https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2>`_)::

   monailabel datasets # List sample datasets
   monailabel datasets --download --name Task09_Spleen --output datasets # Using Task09_Spleen as an example

Start the server
================

Go to the location with the :file:`apps` and :file:`datasets` folders and start the server using::

   monailabel start_server --app apps/deepedit --studies datasets/Task09_Spleen/imagesTr

The server will be served by default at http://127.0.0.1:8000/. It uses `Uvicorn <https://www.uvicorn.org/>`_, which is an ASGI server implementation using `uvloop <https://github.com/MagicStack/uvloop>`_ and `httptools <https://github.com/MagicStack/httptools>`_.

3D Slicer plugin
================

| GitHub: https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/slicer
| Demo: https://youtu.be/o8HipCgSZIw

The pluging should be installed in 3D Slicer from -> Extension Manager -> Active Learning -> MONAI Label

Once installed, the module will be found in the Modules dropdown under Active Learning -> MONAILabel. Input the server address (http://127.0.0.1:8000/) then pick the app name and volume.

Developer model
---------------

If you get an error that you need to upgrade, you need to install it in developer mode.

Uninstall the non-working MONAI Label extension from 3D Slicer and restart the application.

Go to the location you want to download the MONAI Label source code and run::

    git clone git@github.com:Project-MONAI/MONAILabel.git

If you get a message you were denied permission, you need to make a SSH key.

Go to your :file:`~/.ssh` folder and generate a key::

    cd ~/.ssh && ssh-keygen

Keep pressing :kbd:`Enter` until you get the files :file:`id_rsa` and :file:`id_rsa.pub`. On GitHub, go to Account settings > SSH and GPG keys

Run :code:`ssh -T git@github.com` for authentication, then return to the location do download the MONAI Label source code and run :code:`git clone git@github.com:Project-MONAI/MONAILabel.git`.

In 3D Slicer go to Edit -> Application Settings -> Modules -> Additional Module Paths and add a new module path: :file:`<FULL_PATH>/plugins/slicer/MONAILabel`. You can drag and drop this folder into 3D Slicer instead of typing it out. Restart 3D Slicer.