==================
Official Tutorials
==================

MONAI tutorials can be found `on the GitHub page <https://github.com/Project-MONAI/tutorials>`_.

These tutorials require Jupiter Notebooks::

    python -m pip install -U notebook

Some tutorials require optional dependencies, which should have already been installed. In case you get import errors, run::

    pip install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-dev.txt

Run the notebooks from Colab

Data

You should keep all your datasets organized so you don't end up downloading the same datasets again. The following tutorials rely on the environment variable :code:`MONAI_DATA_DIRECTORY` to find the path of your datasets.
Make a :file:`data` folder in your :file:`MONAI` folder. Add it as an environment variable to :file:`bashrc` by running :code:`nano ~/.bashrc` and adding :code:`export MONAI_DATA_DIRECTORY=~/MONAI/data` to the end. Apply the change using :code:`source ~/.bashrc`. This allows you to save results and reuse downloads.
