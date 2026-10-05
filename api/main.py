"""FastAPI server for CyberPulse."""

from fastapi import FastAPI, UploadFile, File
import os
import shutil

from src.inference.predict import predict

app = FastAPI(title="CyberPulse API")

@app.post("/predict")
async def do_prediction(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = predict(temp_path)
        return result
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
@app.get("/health")
def health_check():
    return {"status": "ok"}
