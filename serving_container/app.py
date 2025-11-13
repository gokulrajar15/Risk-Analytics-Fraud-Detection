"""
Simple, optimized FastAPI inference server for fraud detection
"""
import os
import logging
import uvicorn
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import xgboost as xgb
from google.cloud import storage
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

credentials = service_account.Credentials.from_service_account_file("")

model = None

class PredictRequest(BaseModel):
    instances: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[float]

def load_model():
    """Load model once at startup"""
    try:
        model_dir = os.path.join(os.path.dirname(__file__), "model")
        os.makedirs(model_dir, exist_ok=True)

        bucket_name = os.getenv("BUCKET_NAME")
        folder_path = os.getenv("BUCKET_MODEL_FOLDER_PATH")
        model_file_name = "model.bst"
        
        model_path = os.path.join(model_dir, model_file_name)
        if not os.path.exists(model_path):
            client = storage.Client(credentials=credentials)

            bucket = client.bucket(bucket_name)
            blob = bucket.blob(f"{folder_path}/{model_file_name}")
            blob.download_to_filename(model_path)
        

        model = xgb.Booster()
        model.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    yield
    model = None

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    # Convert to numpy and predict
    data = np.array(request.instances, dtype=np.float32)
    dmatrix = xgb.DMatrix(data)
    predictions = model.predict(dmatrix)

    return {"predictions": predictions.tolist()}

@app.get("/")
async def root():
    return {"message": "Fraud Detection API", "status": "ready"}

if __name__ == "__main__":
    uvicorn.run(app, port=8080)