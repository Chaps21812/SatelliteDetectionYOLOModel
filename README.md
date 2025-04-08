# SatelliteDetectionYOLOModel
Satellite Detection code for a containerized YOLO model to detect satellites

Docker Image is based upon the Iron Bank [CUDA+Pytorch](https://repo1.dso.mil/dsop/nextgen-federal/mistk/mistk-python-cuda-pytorch) image. If you are going to build this image, please sign into your account at [Registry1](https://registry1.dso.mil/account/sign-in?redirect_url=%2Fharbor%2Fprojects) through the docker CLI. This will allow you to build the image

## Inference
In order to perform standard inference and model loading, you can simply use the command
```
#Without GPU support
docker run -p 30501:30501 satdetection
#With GPU support
docker run --gpus all -p 30501:30501 satdetection
```

Once your container is running, you can test API endpoints at http://0.0.0.0:30501/docs and use them as a template for standard usage. This container has implemented the following commands:

- /gpu/ - Shows access to GPU for usage in training and inference
- /inference/ - Inferences images based upon json input
- /train/ - Not implemented
- /save/ - Saves the current model and returns it to user
- /load/ - Loads a .pt pytorch model file for inference
- /new/ - Creates a new base instance of a pytorch model.  

## Training
In order to perform training (to avoid poor performance and connection issues) please mount your training data using the command (training is not yet supported)
```
docker run -v /your/path/to/data:/home/python/data satellite_detector
```
To train data, your data must be formatted as a COCO object detection dataset with a file structure below:
dataset/ ├── images/ │ ├── img001.jpg │ ├── img002.jpg │ └── ... ├── labels/ │ ├── img001.txt │ ├── img002.txt │ └── ... └── classes.txt
And annotations must be formatted in the following format: 

## For Developers
To hasten development of containers, remove the copy command in the docker container and use a bind mount for quick changes. Replace the copy file once you are done developing the container. 



