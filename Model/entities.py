from pydantic import BaseModel
from typing import Optional, List

__all__ = ["FitsFile", "ObjectDetection", "ObjectDetections"]


class FitsFile(BaseModel):
    file: str
    sequenceId: Optional[int] = None
    imageSetId: Optional[str] = None
    sequenceCount: Optional[int] = None
    imageSetLength: Optional[int] = None


class ObjectDetection(BaseModel):
    class_id: int
    pixel_centroid: List[float]
    pixel_fwhm: Optional[float] = None
    prob: float
    snr: float
    x_max: float
    x_min: float
    y_max: float
    y_min: float


class ObjectDetections(BaseModel):
    detections: list[ObjectDetection]
