=====
Clara
=====

| Homepage: https://developer.nvidia.com/clara 
| Forums: https://forums.developer.nvidia.com/c/healthcare/149

Clara is an application framework optimized for healthcare and life sciences developers. 

Clara is used in medical devices (`Holoscan <https://developer.nvidia.com/clara-holoscan-sdk>`_), drug discovery (`Discovery <https://www.nvidia.com/en-us/clara/drug-discovery/>`_), smart hospitals (`Guardian <https://developer.nvidia.com/clara-guardian>`_), medical imaging (`Imaging <https://developer.nvidia.com/clara-medical-imaging>`_), and genomics (`Parabricks <https://www.nvidia.com/en-us/clara/genomics/>`_).
We're interested in the medical imaging aspect, especially `Clara Train SDK <https://docs.nvidia.com/clara/clara-train-sdk/index.html>`_ for AI model training and `AIAA <https://docs.nvidia.com/clara/clara-train-sdk/aiaa/index.html>`_ (AI-Assisted Annotation) for AI-Assisted Image Labeling.

CLARA is run through Docker. 

“Clara Train requires Linux having been designed on Ubuntu, and Windows is not a supported platform for Clara Train.” 

Clara Deploy 

| https://docs.nvidia.com/clara/deploy/index.html 
| https://ngc.nvidia.com/catalog/resources/nvidia:clara:clara_bootstrap 
| An extensible reference development framework that facilitates turning AI models into AI-powered clinical workflows with built-in support for DICOM communication and the ability to interface with existing hospital infrastructures.

Clara Studio: https://www.youtube.com/watch?v=ZOfruuXMjkw
Nvidia and Fovia: https://www.youtube.com/watch?v=bllg2lwSfO4

Clara Train SDK Installation
============================
Official installation guide: https://docs.nvidia.com/clara/clara-train-sdk/pt/installation.html

* Their guide has separate instructions for DGX systems, which are supercomputers for problem-solving of machine learning and AI. If you are following my guide, you likely don't have a DGX system.

Download the docker container:

.. code-block:: bash

   export dockerImage=nvcr.io/nvidia/clara-train-sdk:v4.0 
   docker pull $dockerImage 

Create a :file:`/opt/nvidia` folder then run:

.. code-block:: bash

   docker run -it --rm --gpus all --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host --net=host --mount type=bind,source=/opt/nvidia,target=/workspace/data $dockerImage /bin/bash 

.. tip::
   Make sure the command has :code:`--gpus all` or :code:`--gpus=1`. This option is not in the online guide.

Make a :file:`clara-exeriments` folder in your user directory. You can access this local directory in docker if you mount it:

.. code-block:: bash

   docker run --gpus all --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v /home/<username>/clara-experiments:/workspace/clara-experiments $dockerImage /bin/bash 

Replace :code:`<username>` with your username.

:file:`/bin/bash` lets you open a terminal in the container. You can replace that command with something like :code:`nvidia-smi` to make the container run something else while staying in your own terminal. 

.. “MOFED driver not detected. NVIDIA driver not detected.” 