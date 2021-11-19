=============================
AIAA (AI-Assisted Annotation)
=============================

| Homepage: https://docs.nvidia.com/clara/clara-train-sdk/aiaa/index.html 
| Source code: https://github.com/NVIDIA/ai-assisted-annotation-client 

| 3D Slicer plugin: https://github.com/NVIDIA/ai-assisted-annotation-client/tree/master/slicer-plugin 
| https://discourse.slicer.org/t/ai-assisted-segmentation-extension/9536 

MITK plugin 

AIAA Installation
=================

Official installation guide: https://docs.nvidia.com/clara/clara-train-sdk/aiaa/index.html

Create a folder for the AIAA to save models, logs and configurations::

   mkdir aiaa_experiments
   # change the permission because AIAA is running as non-root
   chmod 777 aiaa_experiments

Install Docker Compose
----------------------

Official installation guide: https://docs.docker.com/compose/install/

Download Docker Compose by running the following, replacing :code:`1.29.2` with the latest stable release::
    
   sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose 

Apply executable permissions::
    
   sudo chmod +x /usr/local/bin/docker-compose 

Test installation::
    
   docker-compose --version 

.. tip::
    If :code:`docker-compose` fails, try creating a symbolic link in :file:`/usr/bin`::
        
       sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose 

Set up Docker containers
------------------------

Create files :file:`docker-compose.yml` and :file:`docker-compose.env` by copying the contents from `here <https://docs.nvidia.com/clara/clara-train-sdk/aiaa/quickstart.html#running-aiaa>`_.

In :file:`docker-compose.env`, change :file:`<YOURK WORKSPACE>` to the absolute path of the :file:`aiaa_experiments` folder. 

:file:`AIAA_PORT` can be changed to the port you want to use on the host machine. 

You can modify the :code:`device_ids` section under the :code:`deploy` section of :code:`tritonserver` service to change the GPU id that you want to use. 

Start the AIAA Server
---------------------

Start::

   docker-compose --env-file docker-compose.env -p aiaa_triton up --remove-orphans -d 

Stop::
    
   docker-compose --env-file docker-compose.env -p aiaa_triton down --remove-orphans 

Check log::
    
   docker-compose --env-file docker-compose.env -p aiaa_triton logs -f -t 

Start one service::
    
   docker-compose --env-file docker-compose.env -p aiaa_triton up [the service name] 

Check if AIAA and Triton is running::
    
   docker ps 

Access the server at http://127.0.0.1:$AIAA_PORT, where :code:`$AIAA_PORT` is the port number in :file:`docker-compose.env`.