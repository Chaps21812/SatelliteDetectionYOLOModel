from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from Model.YOLO import YOLO_Satellite_Detection
from Model.entities import ObjectDetections, FitsFile
import logging
import json

from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
model = YOLO_Satellite_Detection()

@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "FastAPI in Docker is running!"}

@app.post("/inference")
async def upload_files(
    data: List[str | FitsFile],
) -> List[ObjectDetections]:
    global model
    global logger
    logger.info(f"Recieved {len(data)} images")
    logger.info(f"Peforming inference...")
    data = [FitsFile(**json.loads(obj)) if isinstance(obj, str) else obj for obj in data]
    results = await model.inference(data)
    logger.info(f"Inference Complete: ")
    return results

@app.post("/inference/")
async def upload_files(
    data: List[str | FitsFile],
) -> List[ObjectDetections]:
    global model
    global logger
    logger.info(f"Recieved {len(data)} images")
    logger.info(f"Peforming inference...")
    data = [json.loads(obj) if isinstance(obj, str) else obj for obj in data]
    results = await model.inference(data)
    logger.info(f"Inference Complete: ")
    return results

if __name__ == "__main__":
    model = YOLO_Satellite_Detection()
