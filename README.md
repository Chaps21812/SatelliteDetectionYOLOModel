# SatelliteDetectionYOLOModel
Satellite Detection code for a containerized YOLO model to detect satellites

Docker Image is based upon the Iron Bank [CUDA+Pytorch](https://repo1.dso.mil/dsop/nextgen-federal/mistk/mistk-python-cuda-pytorch) image. 


In order to perform standard inference and model loading, you can simply use the command

'''

'''


In order to perform training (to avoid poor performance and connection issues) please mount your training data using the command 
'''
docker run -v /your/path/to/data:/home/python/data satellite_detector
'''
To train data, your data must be formatted as a COCO object detection dataset with a file structure below:
dataset/ ├── images/ │ ├── img001.jpg │ ├── img002.jpg │ └── ... ├── labels/ │ ├── img001.txt │ ├── img002.txt │ └── ... └── classes.txt
And annotations must be formatted in the following format: 
'''

'''
