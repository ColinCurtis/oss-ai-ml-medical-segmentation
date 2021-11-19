======
Python
======

Homepage: https://www.python.org/ 

Python is usually automatically installed in Linux distributions.

To run Python, type :code:`python3`. If you want to use :code:`python3` when you type :code:`python`, just add :code:`alias python='python3'` to your :file:`~/.bash_aliases` file.

NumPy
=====
Homepage: https://numpy.org/ 

Requires Python and `pip <https://pypi.org/project/pip/>`_. If you don’t have pip, install using:

.. code-block:: bash

   sudo apt install python3-pip 

.. note::
    If pip requires entering :code:`pip3` instead of :code:`pip`, add :code:`alias pip='pip3'` to your :file:`~/.bash_aliases` file. 

Install NumPy with:

.. code-block:: bash

    pip install numpy

If it tells you to add a directory to PATH, add it to path by adding to :file:`~/.bashrc`:  :code:`PATH=$PATH:/usr/local/bin`, replacing :file:`/usr/local/bin` with the directory the terminal tells you to. 

.. tip::
   Alternatively, instead of opening :file:`bashrc`, you can run :code:`echo "PATH=\$PATH:/usr/local/bin" >> ~/.bashrc` after replacing :file:`/usr/local/bin`.

PyTorch
======= 

| Homepage: https://pytorch.org/ 
| Source code: https://github.com/pytorch/pytorch 

Requirements:

* NVIDIA CUDA 9.2 or above 
* NVIDIA cuDNN v7 or above (?) 
* It is recommended that you use Python 3.6, 3.7 or 3.8

The PyTorch website recommends installing PyTorch with `Anaconda <https://www.anaconda.com/>`_, but pip is good.

From https://pytorch.org/get-started/locally/, select *Stable*, *Linux*, *Pip*, *Python*, and *CUDA 11.1* (or most recent). Run the command given. 

*   If the given comand returns an error, try :code:`pip` instead of :code:`pip3`. 

Verify PyTorch installed correctly by running :code:`python` in the terminal and entering::

   import torch; x = torch.rand(5, 3); print(x) 

The output should be similar to::

   tensor([[0.3380, 0.3845, 0.3217], 
           [0.8337, 0.9050, 0.2650], 
           [0.2979, 0.7141, 0.9069], 
           [0.1449, 0.1132, 0.1375], 
           [0.4675, 0.3947, 0.1426]]) 

Check if your GPU driver and CUDA is enabled and accessible by PyTorch::

   import torch; print(torch.cuda.is_available())

*   Should return :code:`True` 

.. tip::
   You can run this code in one line without starting python by running :code:`python -c "import torch; print(torch.cuda.is_available())"`