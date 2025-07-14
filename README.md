# Single Frame Sentinel
This code is the docker container for the Single Frame Sentinel model used for satellite detection in Machina. This container is capable of deploying any torchscript model to machina.

Docker Image is based upon the Iron Bank [CUDA+Pytorch](https://repo1.dso.mil/dsop/nextgen-federal/mistk/mistk-python-cuda-pytorch) image. If you are going to build this image, please sign into your account at [Registry1](https://registry1.dso.mil/account/sign-in?redirect_url=%2Fharbor%2Fprojects) through the docker CLI. This will allow you to build the image

## Loading Model
Currently, only torchscript models are accepted as inference models. First you need to ensure your model has been created via torch.jit.script or torch.jit.trace in order to have a valid model. Once a model is loaded, place your model in the /model/weights/ directory and rename your model "weights.torchscript". 

Your input schema for the input for your torchscript models should be a torch.tensor [B, C, H, W], where B is batch size, C is Channels, H is Height and W is width. For output, your model return a torch.tensor shaped [BATCH_NUM, 5, NUM_BOXES], where the middle dimension is [centroid_x, centroid_y, box height, box width, box confidence] with unnormalized dimensions for each image. 

Preprocessing fucntions obey the schema [C, H, W] with numpy arrays. To change the preprocessing functions, change the code to use different functions inside the preprocessing python file.

## Inference

To perform inference, raw FITS images must be converted to a b64 encoded string in a list and sent to the /inference/ rest point ([b64_endoded_image, ...]).  These fits images can be of any resolution, but make sure you send the proper images to the proper model. Telescope A images sent to a model trained on Telescope B will yeild poor detections. Your output schema will be a list of detections of the form: 

```
[{detections: [
    {"class_id": int,
    "pixel_centroid": [x_center, y_center],
    "prob": float,
    "snr": float,
    "x_max": float,
    "x_min": float,
    "y_max": float,
    "y_min": float},
    ...]}, 
    ...
]

```

GPU support is enabled for this project. If you want to enable GPU usage, add the gpus argument when running the container. Otherwise, default behavior uses the CPU. 

```
#With GPU support
docker run --gpus all <Container_ID>
```

## Authors and acknowledgment
Created by David Chaparro (david.chaparro@e-o.solutions), Kevin Phan (kevin.phan@e-o.solutions), and Enrique De Alba (enrique.dealba@e-o.solutions) at EO Solutions

# Ironbank
## mistk-python-cuda-pytorch
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
