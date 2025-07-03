from typing import Any, cast
import pydantic
import torch
import base64
import io
import numpy as np
from fastapi.responses import StreamingResponse
from astropy.io import fits
from . import entities
from .settings import settings
from .preprocess import preprocess_image, channel_mixture_C
from . import weights
from importlib import resources

import torchvision
from numpy import typing as npt
import numpy as np
import pathlib

import logging

logger = logging.getLogger(__name__)

class TorchScript_Satellite_Detection:
    def __init__(self) -> None:
        global logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.jit.ScriptModule | None = None  # type: ignore[no-any-unimported]
        self.data_path = pathlib.Path()
        self.load_local()
        logger.info(f"Model is using: {self.device}")
        logger.info(f"Cuda Available: {torch.cuda.is_available()}")

    async def inference(
        self, data: list[str]
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
        x_resolutions = []
        y_resolutions = []

        for i, file in enumerate(data):
            decoded = base64.b64decode(file["file"])
            tempfits = fits.open(io.BytesIO(decoded))
            fitfile = tempfits[0]
            header = fitfile.header
            img_data = fitfile.data
            y_resolutions.append(img_data.shape[0])
            x_resolutions.append(img_data.shape[1])
            if header["TRKMODE"] == "sidereal":
                sidereal_detections += 1
                continue

            arr_float = channel_mixture_C(img_data)  # Expects [C, H, W] float32
            images.append(arr_float)
            rate_indices.append(i)

        images = np.stack(images, axis=0)
        batch = torch.from_numpy(images).to(self.device)
        temp_results = self.model.predict(batch)
        for k, (orig_i, result) in enumerate(zip(rate_indices, temp_results)):
            boxes = result.boxes  # Bounding box object
            detections = []
            x_resolutions[k]
            y_resolutions[k]

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
                            float((xmax.cpu().item() + xmin.cpu().item()) / 2)/x_resolutions[k],
                            float((ymax.cpu().item() + ymin.cpu().item()) / 2)/y_resolutions[k],
                        ],
                        "prob": float(confidence.cpu().item()),
                        "snr": float(signal / noise),
                        "x_max": float(xmax.cpu().item())/x_resolutions[k],
                        "x_min": float(xmin.cpu().item())/x_resolutions[k],
                        "y_max": float(ymax.cpu().item())/y_resolutions[k],
                        "y_min": float(ymin.cpu().item())/y_resolutions[k],
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
            # Load the TorchScript model
            p = resources.files(weights) / settings.MODEL_WEIGHTS
            self.model = torch.jit.load(str(p))  # cast to string if pathlib.Path
            self.model.eval()  # Ensure it's in inference mode
            logger.info(f"TorchScript model loaded successfully from {p}.")
        except Exception as e:
            logger.error(f"Error loading TorchScript model: {type(e).__name__} - {e}")


if __name__ == "__main__":
    model = TorchScript_Satellite_Detection()
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
