# Fraud Detection Serving Container

A custom FastAPI-based serving container for XGBoost fraud detection models on Google Cloud Vertex AI.

## 🎯 Overview

This serving container provides a lightweight, production-ready API for serving fraud detection predictions. It's designed to work with Vertex AI's custom container requirements and supports both local testing and cloud deployment.

## 📁 Files

- `app.py` - Main FastAPI application
- `Dockerfile` - Container definition
- `requirements.txt` - Python dependencies
- `build_and_push.ps1` - PowerShell build script
- `build_and_push.sh` - Bash build script
- `test_api.py` - API testing script

## 🚀 Quick Start

### 1. Build and Push Container

**Windows (PowerShell):**
```powershell
.\build_and_push.ps1
```

**Linux/Mac:**
```bash
chmod +x build_and_push.sh
./build_and_push.sh
```

### 2. Test Locally

```bash
# Run container locally
docker run -p 8080:8080 -e AIP_HTTP_PORT=8080 your-image-uri

# Test API
python test_api.py
```

## 📡 API Endpoints

### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy"
}
```

### Prediction
```http
POST /predict
Content-Type: application/json
```

Request format (Vertex AI compatible):
```json
{
  "instances": [
    [15.724799, 1.875906, 11.009366, 1, 1, 0, 1]
  ]
}
```

Response:
```json
{
  "predictions": [0.02]
}
```

## 🔧 Configuration

### Environment Variables

- `AIP_HTTP_PORT` - Server port (default: 8080)
- `AIP_STORAGE_URI` - Model artifacts location (set by Vertex AI)
- `GOOGLE_APPLICATION_CREDENTIALS` - GCP service account key

### Model Loading

The container supports two model loading methods:

1. **Cloud Storage** (Production): Model loaded from `AIP_STORAGE_URI`
2. **Local File** (Development): Model loaded from `/app/model.bst`

## 🐳 Docker Commands

### Build Image
```bash
docker build -t fraud-detection-serving .
```

### Run Locally
```bash
docker run -p 8080:8080 \
  -e AIP_HTTP_PORT=8080 \
  -v /path/to/model.bst:/app/model.bst \
  fraud-detection-serving
```

### Test Container
```bash
curl -X GET http://localhost:8080/health

curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[15.724799, 1.875906, 11.009366, 1, 1, 0, 1]]}'
```

## ⚙️ Vertex AI Deployment

### 1. Upload Model with Custom Container

```python
from google.cloud import aiplatform

# Upload model with custom serving container
model = aiplatform.Model.upload(
    display_name="fraud-detection-model",
    artifact_uri="gs://your-bucket/model-artifacts/",
    serving_container_image_uri="your-image-uri",
    serving_container_health_route="/health",
    serving_container_predict_route="/predict",
    serving_container_ports=[8080]
)
```

### 2. Deploy to Endpoint

```python
endpoint = aiplatform.Endpoint.create(display_name="fraud-detection-endpoint")

deployed_model = endpoint.deploy(
    model=model,
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=3
)
```

## 🧪 Testing

### Local Testing
```bash
# Install test dependencies
pip install requests google-cloud-aiplatform

# Run tests
python test_api.py
```

### Load Testing
```bash
# Install load testing tool
pip install locust

# Run load test
locust -f load_test.py --host=http://localhost:8080
```

## 📊 Performance

- **Startup Time**: ~30 seconds
- **Prediction Latency**: <50ms
- **Memory Usage**: ~500MB
- **CPU Usage**: Low (optimized for inference)

## 🔍 Monitoring

### Health Checks
- Kubernetes liveness probe on port 8080
- Custom health endpoint at `/health`
- Model loading validation

### Logging
- Structured JSON logs
- Request/response tracking
- Error handling and reporting

## 🔒 Security

- Minimal container image (Python slim)
- No root user execution
- Secure model artifact loading
- Input validation and sanitization

## 🐛 Troubleshooting

### Common Issues

**Model Loading Fails:**
- Check `AIP_STORAGE_URI` environment variable
- Verify GCP credentials
- Ensure model file exists and is accessible

**Health Check Fails:**
- Confirm server is listening on correct port
- Check if model is properly loaded
- Verify container has sufficient resources

**Prediction Errors:**
- Validate input format matches training data
- Check feature count and data types
- Review application logs for details

### Debug Mode
```bash
# Run with debug logging
docker run -e LOG_LEVEL=DEBUG your-image-uri
```

## 📚 Dependencies

- **FastAPI**: Web framework
- **XGBoost**: ML model inference
- **NumPy**: Numerical computing
- **Google Cloud Storage**: Model artifact loading
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

## 🔗 Integration

This serving container integrates with:
- Google Cloud Vertex AI
- Vertex AI Model Registry
- Vertex AI Endpoints
- Google Cloud Storage
- Cloud Monitoring & Logging