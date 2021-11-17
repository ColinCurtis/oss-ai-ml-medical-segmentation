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
or name viewable using ``docker ps``:

.. code-block:: bash
   
   docker exec -t -I mycontainer /bin/bash 


Loading Models
==============

CodeMaster
----------

Robot Turtles
-------------

Primo / Cubetto
---------------