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

   * You should partition at least 300 gigabytes for Linux so you can download large datasets

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
==================

Terminal tutorial: https://www.howtogeek.com/140679/beginner-geek-how-to-start-using-the-linux-terminal/ 

Reboot Linux to BIOS::
   
   sudo systemctl reboot --firmware-setup 

Keep laptop on when closed (two methods)
----------------------------------------

Preferred: Install and use :code:`Tweaks`

Alternative: Run :code:`sudo gedit /etc/systemd/logind.conf` and change :code:`#HandleLidSwitch=suspend` to :code:`#HandleLidSwitch=ignore`

Use BitLocker Flash Drives on Linux 
-----------------------------------

Guide: https://www.ceos3c.com/open-source/open-bitlocker-drive-linux/ 

First-time steps 
~~~~~~~~~~~~~~~~

Install Dislocker::

   sudo apt-get install dislocker 

Make a folder :file:`/media/bitlocker` and :file:`/media/mount`

Steps every time
~~~~~~~~~~~~~~~~

Find your drive::

   sudo fdisk -l

.. note::
   Take note of the device. :file:`/dev/sdb1` would be :file:`sdb1`

Use :code:`Dislocker` to unlock the device, replacing :code:`sdb1` with your device and :code:`YourPassword` with the password::

   sudo dislocker -r -V /dev/sdb1 -uYourPassword -- /media/bitlocker 

Mount the drive::

   sudo mount -r -o loop /media/bitlocker/dislocker-file /media/mount

Move additional disk space to Linux partition
---------------------------------------------

You can move additional storage space from your Windows partition to Linux if you find you need to more space. Although you are unlikely to lose data if you do this correctly, you should still back up anything important before doing these steps. You will want change Windows partition sizes while running Windows and Linux partitions while running Linux.

#. In Windows, go to Disk Management, right-click on the partition to shrink, and select Shrink Volume.
#. If you don't have the Ubuntu flash drive anymore, create a new one then boot into it. 
#. When running Ubuntu from the live flash drive, open GParted and select the relavent disk drive. This is a good guide for using GParted: https://www.howtogeek.com/114503/how-to-resize-your-ubuntu-partitions/
#. In GParted, you can grow partitions into unallocated partitions they are next to.

   * If the target Linux partition is not right next to unallocated space, you will need to move the space one partition at a time until you get to the target partition
   * You can't change sizes of partitions with key logos on them, so in that case you can right-click and select Swapoff
   * If you move the start sector of a partition the OS may fail to boot and you will need to reinstall Grub 2, but I did not have this problem after moving start sectors

#. Restart the computer and verify both operating systems boot correctly