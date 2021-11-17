=====================
Hardware Requirements
=====================

GPU Compatibility
=================

Clara 4.0 is based on the `NVIDIA container for PyTorch release 21.02 <https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-02.html#rel_21-02>`_ detailed `here <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_, which is based on NVIDIA CUDA 11.2.0, which requires NVIDIA Driver release 460.27.04 or later. 

NVIDIA GPUs are ranked by their compute capability. Hardware generations are currently Kepler (3.x), Maxwell (5.x), Pascal (6.x), Volta (7.x), Turing (7.5), and NVIDIA Ampere GPU Architecture (8.0). 

Release 21.02 supports CUDA compute capability 6.0 and higher. Any GPU less than 6.0 will not be compatible. 

View your GPU’s compute capability `here <https://developer.nvidia.com/cuda-gpus>`_.

Example GPUs: 

* GeForce GTX 850M – 5.0
* GeForce GTX 1070 – 6.1
* GeForce GTX 1080 – 6.1