"""
Simple FastAPI serving container for XGBoost fraud detection model
"""
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API", version="1.0.0")


model = None

class PredictRequest(BaseModel):
    instances: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[float]

def load_model():
    """Load the XGBoost model from Cloud Storage or local file"""
    global model
    
    try:
        storage_uri = os.getenv('AIP_STORAGE_URI')
        
        if storage_uri and storage_uri.startswith('gs://'):
            logger.info(f"Loading model from Cloud Storage: {storage_uri}")
            
            bucket_name = storage_uri.replace('gs://', '').split('/')[0]
            blob_path = '/'.join(storage_uri.replace('gs://', '').split('/')[1:])
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(f"{blob_path}/model.bst")
            
            blob.download_to_filename("/tmp/model.bst")
            
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model("/tmp/model.bst")
            logger.info("Model loaded successfully from Cloud Storage")
            
        else:
            # Fallback to local model file
            logger.info("Loading model from local file")
            model_path = "/app/model.bst"
            
            if os.path.exists(model_path):
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(model_path)
                logger.info("Model loaded successfully from local file")
            else:
                raise FileNotFoundError("No model file found")
                
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict endpoint following Vertex AI format"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert instances to numpy array
        instances_array = np.array(request.instances)
        logger.info(f"Received {len(request.instances)} instances for prediction")
        
        # Convert to DMatrix for XGBoost
        import xgboost as xgb
        dmatrix = xgb.DMatrix(instances_array)
        
        # Get predictions (probabilities)
        predictions = model.predict(dmatrix)
        
        # Convert to list for JSON response
        predictions_list = predictions.tolist()
        
        logger.info(f"Generated {len(predictions_list)} predictions")
        return PredictResponse(predictions=predictions_list)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Fraud Detection Model Server", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Vertex AI sets AIP_HTTP_PORT)
    port = int(os.getenv("AIP_HTTP_PORT", 8080))
    
    uvicorn.run(app, host="0.0.0.0", port=port)