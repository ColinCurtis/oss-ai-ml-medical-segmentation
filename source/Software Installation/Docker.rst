======
Docker
======

Installation
============

Official Docker installation guide for Ubuntu: https://docs.docker.com/engine/install/ubuntu/ 

If you have an older version of Docker installed, uninstall them::
    
    sudo apt-get remove docker docker-engine docker.io containerd runc 

Install packages to allow :code:`apt` to use a repository over HTTPS::

   sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release 

Add Docker’s official GPG key::

   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg 

Set up the stable repository::

   echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null 

Install the latest version of Docker Engine and containerd::

   sudo apt-get install docker-ce docker-ce-cli containerd.io 

Verify Docker Engine is installed correctly by running::
    
   sudo docker run hello-world 

To make sure you don’t need to use :code:`sudo` with docker, run this every time::

   sudo groupadd docker
   sudo usermod -aG docker $USER
   newgrp docker

Set up NVIDIA Container Toolkit
===============================

Official guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

Setup the :code:`stable` repository and the GPG key::
   
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list 

Install the :code:`nvidia-docker2` package and dependencies::

   sudo apt-get update 
   sudo apt-get install -y nvidia-docker2 
   sudo systemctl restart docker 

Test by running a base CUDA container::

   sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi 