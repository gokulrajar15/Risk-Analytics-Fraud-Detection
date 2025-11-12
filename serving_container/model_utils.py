"""
Model utilities for saving XGBoost model in the correct format for serving
"""
import os
import json
from pathlib import Path

def save_model_for_serving(model, model_dir: str, model_format: str = "bst"):
    """
    Save XGBoost model in the format expected by the serving container
    
    Args:
        model: Trained XGBoost model
        model_dir: Directory to save the model
        model_format: Format to save model ('bst', 'json', 'ubj')
    """
    # Create model directory if it doesn't exist
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Save model in BST format (recommended for serving)
    if model_format == "bst":
        model_path = os.path.join(model_dir, "model.bst")
        model.save_model(model_path)
        print(f"Model saved in BST format: {model_path}")
    
    elif model_format == "json":
        model_path = os.path.join(model_dir, "model.json")
        model.save_model(model_path)
        print(f"Model saved in JSON format: {model_path}")
    
    elif model_format == "ubj":
        model_path = os.path.join(model_dir, "model.ubj")
        model.save_model(model_path)
        print(f"Model saved in UBJ format: {model_path}")
    
    else:
        raise ValueError(f"Unsupported model format: {model_format}")
    
    # Save model metadata
    metadata = {
        "model_type": "xgboost",
        "format": model_format,
        "filename": f"model.{model_format}",
        "features": model.num_features if hasattr(model, 'num_features') else None,
        "classes": 2  # Binary classification for fraud detection
    }
    
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model metadata saved: {metadata_path}")
    
    return model_path

def load_model_for_testing(model_path: str):
    """
    Load XGBoost model for local testing
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        Loaded XGBoost model
    """
    import xgboost as xgb
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create Booster object and load model
    model = xgb.Booster()
    model.load_model(model_path)
    
    print(f"Model loaded from: {model_path}")
    return model

def test_model_prediction(model, test_instance):
    """
    Test model prediction with a sample instance
    
    Args:
        model: Loaded XGBoost model
        test_instance: List of feature values for prediction
    
    Returns:
        Prediction result
    """
    import xgboost as xgb
    import numpy as np
    
    # Convert test instance to DMatrix
    test_array = np.array([test_instance])
    dmatrix = xgb.DMatrix(test_array)
    
    # Get prediction
    prediction = model.predict(dmatrix)
    
    print(f"Test instance: {test_instance}")
    print(f"Prediction: {prediction[0]}")
    
    return prediction[0]

# Example usage
if __name__ == "__main__":
    # This is just an example - you would replace this with your actual trained model
    print("Model utility functions loaded.")
    print("Use these functions in your training pipeline to save models correctly.")
    
    # Example of how to use in your training script:
    """
    # After training your model
    from model_utils import save_model_for_serving
    
    # Save model for serving
    model_path = save_model_for_serving(
        model=trained_model,
        model_dir="./model_artifacts",
        model_format="bst"
    )
    
    # Test the saved model
    from model_utils import load_model_for_testing, test_model_prediction
    
    loaded_model = load_model_for_testing(model_path)
    test_prediction = test_model_prediction(
        model=loaded_model,
        test_instance=[15.724799, 1.875906, 11.009366, 1, 1, 0, 1]
    )
    """