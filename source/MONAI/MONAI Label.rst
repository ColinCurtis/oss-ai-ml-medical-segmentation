===========
MONAI Label
===========

| GitHub: https://github.com/Project-MONAI/MONAILabel/tree/main/plugins/slicer
| Demo: https://youtu.be/o8HipCgSZIw

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

.. _MONAI-Label-plugin-usage:

3D Slicer plugin
================

:ref:`Install instructions <MONAI-Label-plugin-install>`

