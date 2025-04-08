from fastapi import FastAPI, UploadFile, File
from YOLO import YOLO_Satellite_Detection
import io
import json

app = FastAPI()
model = YOLO_Satellite_Detection()
loaded_model = None

@app.get("/")
def read_root():
    return {"message": "FastAPI in Docker is running!"}

@app.get("/gpu/")
def read_gpu():
    global model
    return model.check_cuda()

@app.post("/inference/")
async def upload_files(file: UploadFile = File(...),  sequenceID: int=1, sequenceLength: int=1):
    global model
    contents = await file.read()  # read as bytes
    text = contents.decode("utf-8")
    images = json.loads(text)

    results = await model.inference(images, sequenceID, sequenceLength)
    return results

@app.post("/train/")
def train(epochs:int,imgsz:int,batch:float):
    global model
    model.train( epochs=1000, imgsz=1200, batch=0.7)
    return {
        "message": "Model Trained successfully"
    }

@app.post("/save/")
def save(model_name:str):
    global model
    saved_response = model.save(model_name)
    return saved_response

@app.post("/load/")
async def upload_model(model_file_pt: UploadFile = File(...)):
    global model
    global loaded_model
    global model
    
    # Read the uploaded file into memory
    contents = await model_file_pt.read()
    buffer = io.BytesIO(contents)
    status = model.load(buffer)

    return status

@app.post("/new/")
def new(size: str="m"):
    global model
    model.new_model(size)
    return {
        "message": "Fresh model created successfully"
    }
