from ultralytics import YOLO
import torch
import base64
from astropy.visualization import ZScaleInterval
import tempfile
import io
import numpy as np
from fastapi.responses import StreamingResponse
from astropy.io import fits
import cv2

class YOLO_Satellite_Detection():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        print(f"Model is using: {self.device}")
        print("Cuda Available:", torch.cuda.is_available())

    def check_cuda(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {"message": f"Model is using: {self.device} \n Cuda Available:{torch.cuda.is_available()}"}

    def train(self, epochs=1000, imgsz=1200, batch=0.7):
        self.model.train(data=self.data_path+"/data.yaml", epochs=epochs,imgsz=imgsz,batch=batch)
        
    def new_model(self, model_size:str="m"):
        if model_size == "n": self.model = YOLO("yolo11n.pt")
        if model_size == "s": self.model = YOLO("yolo11s.pt")
        if model_size == "m": self.model = YOLO("yolo11m.pt")
        if model_size == "l": self.model = YOLO("yolo11l.pt")
        if model_size == "x": self.model = YOLO("yolo11x.pt")

    async def inference(self, data:list, sequenceId=None, imageSetId=None, sequenceCount=None, imageSetLength=None ) -> list:
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        batch_detections = []
        sidereal_detections = 0
        images = []

        for file in data:
            decoded = base64.b64decode(file.file)
            tempfits = fits.open(io.BytesIO(decoded))
            fitfile = tempfits[0]
            header = fitfile.header
            data = fitfile.data
            if header["TRKMODE"]=="sidereal":
                sidereal_detections += 1
                continue

            detections = []
            arr_float = self.preprocess_image(data)
            images.append(arr_float)

        images = np.stack(images, axis=0)
        batch = torch.from_numpy(images).to(self.device)
        temp_results = self.model.predict(batch)
        for k,result in enumerate(temp_results):
            boxes = result.boxes  # Bounding box object
            for b in range(boxes.xyxy.shape[0]):
                (xmin, ymin, xmax, ymax) = boxes.xyxy[b,:].cpu()
                class_id = boxes.cls[b]
                confidence = boxes.conf[b]
                signal = images[k,0,int((xmax-xmin)/2), int((ymax-ymin)/2)]
                noise = np.std(images[k,0,:,:])
                detection = {
                    "class_id":int(class_id.cpu().item()),
                    "pixel_centroid": [float((xmax.cpu().item()+xmin.cpu().item())/2), float((ymax.cpu().item()+ymin.cpu().item())/2)],
                    "pixel_fwhm":"N/A",
                    "prob":float(confidence.cpu().item()),
                    "snr": float(signal/noise),
                    "x_max": float(xmax.cpu().item()),
                    "x_min": float(xmin.cpu().item()),
                    "y_max": float(ymax.cpu().item()),
                    "y_min": float(ymin.cpu().item()),
                }
                detections.append(detection)
            single_image_detection = {"detections": detections}
            batch_detections.append(single_image_detection)
        for i in range(sidereal_detections):
            batch_detections.append({"detections": []})
        return batch_detections
    
    def save(self, save_name:str):
        self.model = self.model.to('cpu')
        # Save model weights and biases (state_dict) to an in-memory buffer
        buffer = io.BytesIO()
        torch.save(self.model, buffer)
        buffer.seek(0)
        self.model = self.model.to(self.device)
        return StreamingResponse(buffer, media_type="application/octet-stream", headers={
        "Content-Disposition": f"attachment; filename={save_name}.pt",
        "message": "Model saved successfully"
        })

    def load(self, load_buffer:io.BytesIO): 
        with tempfile.NamedTemporaryFile(suffix=".pt") as temp_file:
            temp_file.write(load_buffer.read())
            temp_file.flush()
            self.model = YOLO(temp_file.name)
        return {"message": "Model loaded successfully"}
    
    def load_local(self): 
        try:
            # Load the model
            self.model = YOLO("/home/python/Model/model.pt")
            return {"message": "Model loaded successfully from /home/python/Model/model.pt"}
        except Exception as e:
            return {"message": "Error loading model"}

    def preprocess_image(self, image:np.ndarray):
        # Apply zscale to the image data for contrast enhancement
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(image)
        
        # Apply Z-scale normalization (clipping values between vmin and vmax)
        image = np.clip(image, vmin, vmax)
        image = (image - vmin) / (vmax - vmin) * 255  # Scale to 0-255 range
        
        # Convert the image data to an unsigned 8-bit integer (for saving as PNG)
        image = image.astype(np.float32)

        height, width = image.shape
        new_height = (height // 32) * 32 if height % 32 == 0 else ((height // 32) + 1) * 32
        new_width = (width // 32) * 32 if width % 32 == 0 else ((width // 32) + 1) * 32
        resized_image = cv2.resize(image, (new_width, new_height))

        image = np.stack([resized_image] * 3, axis=0)
        return image
            
            

if __name__ == "__main__":
    model = YOLO_Satellite_Detection()
    model.new_model()
    test_list = []
    with open("/mnt/c/Users/david.chaparro/Documents/Repos/SatelliteDetectionYOLOModel/test_image.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        for i in range(3):
            my_dict = {"file": encoded_string,
                       "sequence_ID":i,
                       "sequenceLength": len(encoded_string)}
            test_list.append(my_dict)
        results = model.inference(test_list)

    print("Done!")

    
