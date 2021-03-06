===========
Nvidia CUDA
===========

:abbr:`CUDA (Compute Unified Device Architecture)` is a parallel computing platform and :abbr:`API (Application Programming Interface)` model that allows developers to use CUDA-enabled :abbr:`GPUs (Graphics Processing Unit)` for general purpose processing. CUDA works with all Nvidia GPUs from the G8x series onwards, including GeForce, Quadro and the Tesla line and is compatible with most standard operating systems.

| Homepage: https://developer.nvidia.com/cuda-toolkit 
| Forum: https://forums.developer.nvidia.com/c/accelerated-computing/cuda/206 

| Installations and documentations of current and past versions: https://developer.nvidia.com/cuda-toolkit-archive 
| Release notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html 

| CUDA quick start guide: https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html 
| CUDA for Linux installation guide: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html 

Backwards and Fowards compatibility: https://docs.nvidia.com/pdf/CUDA_Compatibility.pdf, https://docs.nvidia.com/deploy/cuda-compatibility/ 

The CUDA toolkit consists of two main components: 

* The development libraries (including the CUDA runtime)
* The driver components. The driver components are further separated into two categories: 

   * Kernel mode components (the ‘display’ driver)
   * User mode components (the CUDA driver, the OpenGL driver, etc.)

Each release of the CUDA Toolkit requires a minimum version of the CUDA driver. The CUDA driver is backward compatible, meaning that applications compiled against a particular version of the CUDA will continue to work on subsequent (later) driver releases. 

| CUDA11 needs NVIDIA driver >= 450.36.06. 
| CUDA 11.4 Update 2 needs NVIDIA driver >=470.57.02 

The NVIDIA driver can be installed as part of the CUDA Toolkit installation, but you should install it separately.

The CUDA driver on Linux systems is :code:`libcuda.so`.

The minimum recommended GCC compiler (GNU Compiler Collection) is at least GCC 6. 

Unless you are using a supercomputer you probably don’t have a POWER8 or POWER9 system, so ignore those steps. You likely have a x86_64 workstation. Only Tesla V100 and T4 GPUs are supported for CUDA 11.4 on Arm64 (aarch64) POWER9 (ppc64le). 

It's possible to install CUDA on :abbr:`WSL (Windows Subsystem for Linux)`: https://docs.nvidia.com/cuda/wsl-user-guide/index.html

CUDA Installation
=================

The steps in this section are mainly derived or copied straight from `the official CUDA Linux guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_ and this article: https://www.iridescent.io/tech-blogs-installing-cuda-the-right-way/

1. Pre-installation Actions
---------------------------

Check your GPU compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NVIDIA Clara needs a compute capability of 6.0 or higher. Your GPU should be listed here: https://developer.nvidia.com/cuda-gpus. You can find out what GPU you have with:

.. code-block:: bash

   lspci | grep -I NVIDIA   

Verify you have a supported version of Linux 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CUDA Toolkit is only supported on specific Linux distributions, which are listed in the release notes. Find your system information with: 

.. code-block:: bash

   uname -m && cat /etc/*release

x86_64 means you have a 64-bit system, and the remainder is your distribution information 

Install GCC 
~~~~~~~~~~~

Check if you have GCC by running :code:`gcc --version` . It should be at least GCC 6. 

If you don’t have gcc, install it using :code:`sudo apt install gcc`

2. Install NVIDIA drivers
-------------------------

Method 1 (Using package manager as of Ubuntu 19.04)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run :code:`nvidia-smi` to verify your NVIDIA drivers are set up properly. If that doesn’t work, use :code:`modinfo nvidia` to check your GPU.  

If :code:`nvidia-smi` didn’t work, install the latest version of *nvidia-utils* that is recommended in the terminal. For example: :code:`sudo apt install nvidia-utils-470`

You can also check version using :code:`cat /proc/driver/nvidia/version`

Method 2 (Only if method 1 doesn’t work)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search for your GPU driver and download the Production Branch here: https://www.nvidia.com/Download/index.aspx?lang=en-us 

Run :code:`chmod +x <driver file>`, replacing *<driver file>* with the downloaded driver file name.

Reboot the computer and enter a TTY session by pressing :kbd:`Ctrl` + :kbd:`Alt` + :kbd:`F3` or another key combination if you’re not using a recent version of Ubuntu.

*   More about TTY here: https://askubuntu.com/questions/66195/what-is-a-tty-and-how-do-i-access-a-tty 

.. tip::
   Move between a few TTY terminals including the login screen so it’s not as buggy later on (this may be because you need to launch TTY from login screen) 

Kill the X-server using :code:`sudo service lightdm stop`, or if that doesn’t work, kill GDM using :code:`sudo service gdm stop`. 

*   If the screen goes blank, switch to another TTY terminal. If that still doesn’t work, reboot the computer, and try switching between TTY terminals beforehand. 

Change directory to the location of the driver and run :code:`sudo ./NVIDIA-Linux-<system>-<version>.run`

3. Download and install CUDA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the CUDA *runfile (local)* installer for your distribution from https://developer.nvidia.com/cuda-downloads and run :code:`sudo sh <downloaded file>`, where *<downloaded file>* is the downloaded file. If the terminal says it is not executable, run :code:`chmod +x <downloaded file>` first. 

*   The runfile should be easier to use and more reliable than the deb files 

If it says you already have a driver, continue anyways. Accept TOS. Select the Driver checkbox to uncheck it. Install. 

Open your :file:`~/.bashrc` file (:code:`nano ~/.bashrc`), and to the end of the file, append :code:`export PATH=$PATH:/usr/local/cuda/bin` and :code:`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`

Reboot and run :code:`nvcc --version` to test installation success.

*   If the command is not found, you did something wrong, likely at the :file:`bashrc` step. Do not install *nvidia-cuda-toolkit* 

*   You shouldn’t need to reboot; you just need to run :code:`source ~/.bashrc` 