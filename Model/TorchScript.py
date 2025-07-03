from typing import Any, cast
import pydantic
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

class TorchScript_Satellite_Detection:
    def __init__(self) -> None:
        global logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.jit.ScriptModule | None = None  # type: ignore[no-any-unimported]
        self.data_path = pathlib.Path()
        self.load_local()
        logger.info(f"Model is using: {self.device}")
        logger.info(f"Cuda Available: {torch.cuda.is_available()}")

    async def inference(self, data: list[entities.FitsFile]) -> list[entities.ObjectDetections]:
        if self.model is None:
            logger.error("No model loaded")
        assert self.model is not None

        self.model = self.model.to(self.device).eval()

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

            arr_float = self.preprocess_image(img_data)  # Expects [C, H, W] float32
            images.append(arr_float)
            rate_indices.append(i)

        if not images:
            return pydantic.TypeAdapter(list[entities.ObjectDetections]).validate_python(batch_detections)

        images_np = np.stack(images, axis=0)  # Shape: [B, C, H, W]
        batch = torch.from_numpy(images_np).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)  # Should be shape: [B, 5, N]
            logger.info(f"Output shape: {outputs.shape}")

        for k, (orig_i, preds) in enumerate(zip(rate_indices, outputs)):
            detections = []
            preds = preds.permute(1, 0)  # Now shape: [N, 5]
            H, W = images_np[k].shape[1:]  # assume [C, H, W]

            for det in preds:
                x_c, y_c, w, h, conf = det.tolist()
                if conf < 0.5:  # Confidence threshold (adjustable)
                    continue

                # Convert normalized center/size to pixel coordinates
                x_c *= W
                y_c *= H
                w *= W
                h *= H

                xmin = x_c - w / 2
                xmax = x_c + w / 2
                ymin = y_c - h / 2
                ymax = y_c + h / 2

                # Clamp to image bounds
                xmin = max(0, xmin)
                xmax = min(W - 1, xmax)
                ymin = max(0, ymin)
                ymax = min(H - 1, ymax)

                signal = images_np[k, 0, int(y_c), int(x_c)]
                noise = np.std(images_np[k, 0])

                detection = {
                    "class_id": 0,  # Only one class
                    "pixel_centroid": [float(x_c), float(y_c)],
                    "prob": float(conf),
                    "snr": float(signal / noise) if noise > 0 else 0,
                    "x_max": float(xmax),
                    "x_min": float(xmin),
                    "y_max": float(ymax),
                    "y_min": float(ymin),
                }
                detections.append(detection)

            batch_detections[orig_i]["detections"] = detections

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


    def preprocess_image(self, image: npt.NDArray) -> npt.NDArray:
        # Apply zscale to the image data for contrast enhancement
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(image)

        # Apply Z-scale normalization (clipping values between vmin and vmax)
        #image = np.clip(image, vmin, vmax)
        #image = (image - vmin) / (vmax - vmin) * 255  # Scale to 0-255 range
        # Convert the image data to an unsigned 8-bit integer (for saving as PNG)
        
        image = image.astype(np.float32)

        height, width = image.shape
        new_height = (
            (height // 32) * 32 if height % 32 == 0 else ((height // 32) + 1) * 32
        )
        new_width = (width // 32) * 32 if width % 32 == 0 else ((width // 32) + 1) * 32
        #resized_image = cv2.resize(image, (new_width, new_height))
        resized_image = cv2.resize(image, (512, 512))
        image = np.stack([resized_image] * 3, axis=0)
        
        return image


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
