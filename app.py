from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from Model.YOLO import YOLO_Satellite_Detection
from Model.entities import ObjectDetections, FitsFile

import io
from typing import List


app = FastAPI()
model = YOLO_Satellite_Detection()
model.load_local()

loaded_model = None


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "FastAPI in Docker is running!"}


@app.get("/gpu/")
def read_gpu() -> dict[str, str]:
    global model
    return model.check_cuda()


@app.get("/load_local/")
def read_local() -> dict[str, str]:
    global model
    return model.load_local()


@app.post("/inference/")
async def upload_files(
    data: List[FitsFile],
) -> List[ObjectDetections]:
    global model
    results = await model.inference(data)
    print("Im done")
    return results


@app.post("/train/")
def train(epochs: int, imgsz: int, batch: float) -> dict[str, str]:
    global model
    model.train(epochs=1000, imgsz=1200, batch=0.7)
    return {"message": "Model Trained successfully"}


@app.post("/save/")
def save(model_name: str) -> StreamingResponse:
    global model
    saved_response = model.save(model_name)
    return saved_response


@app.post("/load/")
async def upload_model(model_file_pt: UploadFile = File(...)) -> dict[str, str]:
    global model
    global loaded_model
    global model

    # Read the uploaded file into memory
    contents = await model_file_pt.read()
    buffer = io.BytesIO(contents)
    status = model.load(buffer)

    return status


@app.post("/new/")
def new(size: str = "m") -> dict[str, str]:
    global model
    model.new_model(size)
    return {"message": "Fresh model created successfully"}
