=====
Linux
=====

It might be possible to run everything in Windows through Windows 
Subsystem for Linux (WSL), but it might be easier to install and 
troubleshoot a native install of Linux on a computer with an NVIDIA GPU. 
You can dual-boot a computer so you can boot into either Linux or 
Windows natively.

Ubuntu
======

Ubuntu has a large community and it good for beginners of Linux, so 
troubleshooting should be easier than on other distros. Some installation 
instructions have examples specifically for Ubuntu.  Red Hat Enterprise 
Linux (RHEL) also is a very popular distro and has specific examples 
during installation steps.

Some software say they require Ubuntu and specifically a long-term support 
(LTS) version like 20.04. For maximum compatibility, I recommend this 
version.

Install Ubuntu 20.04 :abbr:`LTS (Long Term Support)`
----------------------------------------------------

You can either replace your existing OS following `the steps here <https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview>`_, or you can dual boot with Windows. The following steps are based on `this guide <https://medium.com/linuxforeveryone/how-to-install-ubuntu-20-04-and-dual-boot-alongside-windows-10-323a85271a73>`_, and you may want to reference it for more details.

#. `Download Ubuntu 20.04 LTS <https://ubuntu.com/download#download>`_
#. Use `Rufus <https://rufus.ie/en/>`_ to flash the Linux ISO onto a flash drive 
#. Shrink a partition to get hard drive space for Linux 
#. Restart the computer into the boot menu and boot from the flash drive 

   * If pressing keys at startup doesn’t work, go into Recovery settings in Windows and choose to restart into firmware settings 
   * If possible, disable the fast-boot option from your BIOS 

#. Pick Normal installation, Download updates, and Install third-party software 
#. Choose installation type “Something else” 
#. Select the free space and click the + button to create Linux partitions 

   * If you want SWAP memory, make that first and choose “Logical” and “swap area” 
   * Create the root partition using the rest of the free space. Choose “Primary” and use as “Ext4 journaling file system” and make the Mount point “/” 
   * If you have a separate hard drive for personal files, create a Logical partition using free space on that hard drive 

#. Leave the boot loader installation on /dev/sda or equivalent top-level location that stores Windows Boot Manager then click “Install Now” 
#. After installation if it always boots straight into Windows, you should make the Ubuntu boot loader top priority in the BIOS/UEFI settings 
#. Configure the settings to your liking and update the system and packages using this code in the terminal:

.. code-block:: bash

   sudo -- sh -c 'apt-get update; apt-get upgrade -y; apt-get dist-upgrade -y; apt-get autoremove -y; apt-get autoclean -y'

Useful Ubuntu tips
------------------

Terminal tutorial: https://www.howtogeek.com/140679/beginner-geek-how-to-start-using-the-linux-terminal/ 

Reboot Linux to BIOS::
   
   sudo systemctl reboot --firmware-setup 

Keep laptop on when closed (two methods):
  
* Preferred: Install and use :code:`Tweaks`

* Alternative: Run :code:`sudo gedit /etc/systemd/logind.conf` and change :code:`#HandleLidSwitch=suspend` to :code:`#HandleLidSwitch=ignore`