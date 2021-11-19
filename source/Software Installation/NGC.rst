===================================================================
:abbr:`NGC (NVIDIA GPU Cloud)` :abbr:`CLI (command-line interface)`
===================================================================

Official installation guide: https://ngc.nvidia.com/catalog/collections/nvidia:gettingstarted

Login to your NVIDIA account
============================

Most software does not require an account, but some require an API key from your account 

| Sign in or make an account: https://ngc.nvidia.com/signin 
| Generate an API key: https://ngc.nvidia.com/setup/api-key 

.. note::
    If you lose the API key, you can generate a new one and the old one will be invalidated 

AMD64 Linux NGC CLI Installation
================================

This section follows `this guide <https://ngc.nvidia.com/setup/installers/cli>`_ to installing the AMD64 Linux version of NGC CLI.

Move to the directory you would like to store NGC CLI and run::
    
   wget -O ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_linux.zip && unzip -o ngccli_linux.zip && chmod u+x ngc 

Add current directory to path::
    
   echo "export PATH=\"\$PATH:$(pwd)\"" >> ~/.bashrc && source ~/.bashrc 

The original code appends :file:`.bash_profile`, but only :file:`.bashrc` worked for me.

Run :code:`ngc config set`, enter your API key, enter :code:`ascii`, then enter the choices given until NGC is configured. 