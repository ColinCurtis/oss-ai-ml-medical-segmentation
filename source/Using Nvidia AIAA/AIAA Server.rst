==========
Using AIAA
==========

Startup
=======
Run and start:

.. code-block:: bash

   ./start_aiaa.sh

   docker-compose --env-file docker-compose.env -p aiaa_triton up --remove-orphans -d 

Can also be integrated in segmentation clients (MITK Workbench, 3D Slicer, Fovia (PACS backend), OHIF) 

Connecting to server
====================
Access Docker container files, replacing *mycontainer* with the container id 
or name (viewable using :code:`docker ps`):

.. code-block:: bash
   
   docker exec -t -I mycontainer /bin/bash


Loading Models
==============
https://docs.nvidia.com/clara/clara-train-sdk/aiaa/loading_models.html


View accessible models: `127.0.0.1:5000/v1/models <https://127.0.0.1:5000/v1/models>`_