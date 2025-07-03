from typing import Any, cast
import pydantic
from ultralytics import YOLO
import torch
import base64
from astropy.visualization import ZScaleInterval
import io
import numpy as np
from fastapi.responses import StreamingResponse
from astropy.io import fits
import cv2
from . import entities
from .settings import settings
from . import weights
from importlib import resources

from numpy import typing as npt
import pathlib

import logging

logger = logging.getLogger(__name__)

class YOLO_Satellite_Detection:
    def __init__(self) -> None:
        global logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: YOLO | None = None  # type: ignore[no-any-unimported]
        self.data_path = pathlib.Path()
        self.load_local()
        logger.info(f"Model is using: {self.device}")
        logger.info(f"Cuda Available: {torch.cuda.is_available()}")

    async def inference(
        self, data: list[entities.FitsFile]
    ) -> list[entities.ObjectDetections]:
        if self.model is None:
            logger.error("No model loaded")
        assert self.model is not None
        logger.info(f"Length of inference request: {len(data)}")
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        batch_detections: list[dict[str, list[dict[str, Any]]]] = [
            {"detections": []} for _ in data
        ]
        sidereal_detections = 0
        images = []
        rate_indices = []

        for i, file in enumerate(data):
            decoded = base64.b64decode(file.file)
            tempfits = fits.open(io.BytesIO(decoded))
            fitfile = tempfits[0]
            header = fitfile.header
            img_data = fitfile.data
            if header["TRKMODE"] == "sidereal":
                sidereal_detections += 1
                continue

            arr_float = self.preprocess_image(img_data)
            images.append(arr_float)
            rate_indices.append(i)

        images = np.stack(images, axis=0)
        batch = torch.from_numpy(images).to(self.device)
        temp_results = self.model.predict(batch)
        for k, (orig_i, result) in enumerate(zip(rate_indices, temp_results)):
            boxes = result.boxes  # Bounding box object
            detections = []

            if boxes is not None:
                for b in range(cast(int, boxes.xyxy.shape[0])):
                    xmin, ymin, xmax, ymax = boxes.xyxy[b, :].cpu()
                    class_id = boxes.cls[b]
                    confidence = boxes.conf[b]
                    signal = images[  # type: ignore[call-overload]
                        k, 0, int((xmax - xmin) / 2), int((ymax - ymin) / 2)
                    ]
                    noise = np.std(images[k, 0, :, :])  # type: ignore[call-overload]
                    detection = {
                        "class_id": int(class_id.cpu().item()),
                        "pixel_centroid": [
                            float((xmax.cpu().item() + xmin.cpu().item()) / 2),
                            float((ymax.cpu().item() + ymin.cpu().item()) / 2),
                        ],
                        "prob": float(confidence.cpu().item()),
                        "snr": float(signal / noise),
                        "x_max": float(xmax.cpu().item()),
                        "x_min": float(xmin.cpu().item()),
                        "y_max": float(ymax.cpu().item()),
                        "y_min": float(ymin.cpu().item()),
                    }
                    detections.append(detection)
            batch_detections[orig_i]["detections"] = detections
            logger.info(f"Length of detections: {len(batch_detections)}")
        return pydantic.TypeAdapter(list[entities.ObjectDetections]).validate_python(
            batch_detections
        )

    def save(self, save_name: str) -> StreamingResponse:
        assert self.model is not None
        self.model = self.model.to("cpu")
        # Save model weights and biases (state_dict) to an in-memory buffer
        buffer = io.BytesIO()
        torch.save(self.model, buffer)
        buffer.seek(0)
        self.model = self.model.to(self.device)
        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={save_name}.pt",
                "message": "Model saved successfully",
            },
        )

    def load_local(self) -> dict[str, str]:
        try:
            # Load the model
            p = resources.files(weights) / settings.MODEL_WEIGHTS
            self.model = YOLO(p)
            logger.info(f"Model loaded successfully from {p}")
        except Exception as e:
            logger.error(f"Error loading model: {type(e).__name__} - {e}")

    def preprocess_image(self, image: npt.NDArray) -> npt.NDArray:
        # Apply zscale to the image data for contrast enhancement
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(image)

        # Apply Z-scale normalization (clipping values between vmin and vmax)
        image = np.clip(image, vmin, vmax)
        image = (image - vmin) / (vmax - vmin) * 255  # Scale to 0-255 range

        # Convert the image data to an unsigned 8-bit integer (for saving as PNG)
        image = image.astype(np.float32)

        height, width = image.shape
        new_height = (
            (height // 32) * 32 if height % 32 == 0 else ((height // 32) + 1) * 32
        )
        new_width = (width // 32) * 32 if width % 32 == 0 else ((width // 32) + 1) * 32
        resized_image = cv2.resize(image, (new_width, new_height))

        image = np.stack([resized_image] * 3, axis=0)
        return image


if __name__ == "__main__":
    model = YOLO_Satellite_Detection()
    test_list = []
    with open("/mnt/c/Users/david.chaparro/Documents/Repos/SatelliteDetectionYOLOModel/test_image.png","rb",) as image_file:
        encoded_string = base64.b64encode(image_file.read())
        for i in range(3):
            my_dict = {
                "file": encoded_string,
                "sequence_ID": i,
                "sequenceLength": len(encoded_string),
            }
            test_list.append(my_dict)
        results = model.inference(
            pydantic.TypeAdapter(list[entities.FitsFile]).validate_python(test_list)
        )

    print("Done!")
