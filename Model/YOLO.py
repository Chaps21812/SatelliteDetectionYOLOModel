from ultralytics import YOLO
import torch
import torchvision
import base64
from PIL import Image
import tempfile
import io
import numpy as np
from fastapi.responses import StreamingResponse
from astropy.io import fits

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

    async def inference(self, data:list, sequenceID: int, sequenceLength: int) -> list:
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        batch_detections = []
        images = []

        for file in data:
            decoded = base64.b64decode(file["file"])
            tempfits = fits.open(io.BytesIO(decoded))
            fitfile = tempfits[0]
            header = fitfile.header
            data = fitfile.data
            if header["TRKMODE"]=="sidereal":
                continue

            detections = []
            arr_float = data.astype(np.float32)/65535
            image_torch = torch.from_numpy(arr_float)
            image_torch = torchvision.transforms.Resize(size=(640,640))(image_torch.unsqueeze(0))
            image_torch = image_torch.repeat(3,1,1)
            images.append(image_torch)

        batch = torch.stack(images)
        temp_results = self.model.predict(batch)
        for k,result in enumerate(temp_results):
            boxes = result.boxes  # Bounding box object
            for b in range(boxes.xyxy.shape[0]):
                (xmin, ymin, xmax, ymax) = boxes.xyxy[b,:].cpu()
                class_id = boxes.cls[b]
                confidence = boxes.conf[b]
                signal = batch[k,0,int((xmax-xmin)/2), int((ymax-ymin)/2)]
                noise = np.std(batch[k,0,:,:].cpu().numpy())
                detection = {
                    "class_id":class_id.cpu().item(),
                    "pixel_centroid": [(xmax.cpu().item()-xmin.cpu().item())/2/640, (ymax.cpu().item()-ymin.cpu().item())/2/640],
                    "pixel_fwhm":"N/A",
                    "prob":confidence.cpu().item(),
                    "snr": (signal/noise).cpu().item(),
                    "x_max": xmax.cpu().item()/640,
                    "x_min": xmin.cpu().item()/640,
                    "y_max": ymax.cpu().item()/640,
                    "y_min": ymin.cpu().item()/640,
                }
                detections.append(detection)
            single_image_detection = {"detections": detections}
            batch_detections.append(single_image_detection)

        return batch_detections
    
    def save(self, save_name:str):
        # Save model weights and biases (state_dict) to an in-memory buffer
        buffer = io.BytesIO()
        torch.save(self.model, buffer)
        buffer.seek(0)
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

    
