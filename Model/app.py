from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI in Docker is running!"}

@app.post("/inference/")
def inference(image: str, sequenceID: int, sequenceLength: int):
    
    return results

@app.post("/train/")
def train(image: str, sequenceID: int, sequenceLength: int):
    decoded_image = base64.b64decode(image)  # Decode base64 image
    return {
        "sequenceID": sequenceID,
        "sequenceLength": sequenceLength,
        "message": "Image processed successfully"
    }

@app.post("/save/")
def save(image: str, sequenceID: int, sequenceLength: int):
    decoded_image = base64.b64decode(image)  # Decode base64 image
    return {
        "message": "Model saved successfully"
    }

@app.post("/load/")
def load(image: str, sequenceID: int, sequenceLength: int):
    decoded_image = base64.b64decode(image)  # Decode base64 image
    return {
        "message": "Model loaded successfully"
    }

@app.post("/new/")
def new(image: str, sequenceID: int, sequenceLength: int):
    decoded_image = base64.b64decode(image)  # Decode base64 image
    return {
        "message": "Model loaded successfully"
    }
