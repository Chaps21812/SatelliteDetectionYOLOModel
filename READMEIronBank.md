# mistk-python-cuda-pytorch
### Iron Bank image for Model Integration Software ToolKit (MISTK) including CUDA and PyTorch
This image builds upon the **mistk-python-cuda** [project](https://repo1.dso.mil/dsop/nextgen-federal/mistk/mistk-python-cuda)
and 
[images](https://ironbank.dso.mil/repomap/details;registry1Path=nextgen-federal%2Fmistk%2Fmistk-python-cuda),
adding PyTorch with CUDA support.  

Information about MISTK:
- MISTK Documentation: https://mistkml.github.io/index.html
- MISTK Code: https://github.com/mistkml/mistk/

This image is intended to be used as a base image for MISTK-wrapped Models, Transforms and Evaluators
that require PyTorch with GPU support.

See https://mistkml.github.io/deployment.html for an example of a Dockerfile for a MISTK-wrapped model.  

Docker Image Versioning

The Docker image is versioned using the format mistk_version-python_version-cuda_version-pytorch_version. For example, if the MISTK version is 1.2.0, the Python version is 3.11.8, the Cuda version is 12.3, and the Pytorch version is 2.2.0; the Docker image version would be 1.2.0-3.11.8-12.3-2.2.0

PyTorch GPU access can be tested inside the container like the example below, hopefully producing the names
of the GPUs on your hardware.
```
    $ python
    >>> import torch
    >>> torch.cuda.is_available()
    True
    >>> torch.cuda.current_device()
    0
    >>> torch.cuda.get_device_name(0)
    'Tesla V100-SXM2-16GB'
    >>> torch.cuda.get_device_name(1)
    'Tesla V100-SXM2-16GB'
    >>> torch.cuda.get_device_name(2)
    'Tesla V100-SXM2-16GB'
    >>> torch.cuda.get_device_name(3)
    'Tesla V100-SXM2-16GB'
```
