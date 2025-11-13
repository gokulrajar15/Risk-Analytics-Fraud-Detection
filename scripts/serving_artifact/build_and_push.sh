#!/usr/bin/env bash
set -euo pipefail

# === Variables – adjust as needed ===
PROJECT_ID="${PROJECT_ID:-poetic-velocity-459409-f2}"
REGION="${REGION:-us-central1}"
REPO_NAME="${SERVING_REPO_NAME:-vertex-ai-customservingartifact}"
IMAGE_NAME="${SERVING_IMAGE_NAME:-fraud-detection-serving}"
TAG="${TAG:-latest}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-./serving_container}"

# Full image URI for Artifact Registry
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}"

# === Script starts ===
echo "🔧 Building & Pushing Serving Container to Artifact Registry"
echo "Project: ${PROJECT_ID:0:8}***" # Only show first 8 chars for security
echo "Region: $REGION"
echo "Repository: $REPO_NAME"
echo "Image name: $IMAGE_NAME"
echo "Tag: $TAG"
echo "Image URI: ${REGION}-docker.pkg.dev/${PROJECT_ID:0:8}***/${REPO_NAME}/${IMAGE_NAME}:${TAG}"
echo ""

# Check if required variables are set
if [ "$PROJECT_ID" = "your-gcp-project-id" ]; then
    echo "❌ ERROR: Please set your actual GCP Project ID!"
    echo "Use: export PROJECT_ID=\"poetic-velocity-459409-f2\""
    exit 1
fi

# Function to handle errors
handle_error() {
    echo "❌ ERROR: $1" >&2
    exit 1
}

# Verify gcloud authentication
verify_auth() {
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        handle_error "No active gcloud authentication found. Please run 'gcloud auth login' first."
    fi
}

# Test container health
test_container_health() {
    local container_id=$1
    echo "🔍 Testing container health..."
    
    # Wait for container to start
    sleep 10
    
    # Test health endpoint
    if curl -f -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ Health check passed"
    else
        echo "⚠️  Health check warning (container may still be starting)"
    fi
    
    # Clean up test container
    docker stop "$container_id" > /dev/null 2>&1 || true
    docker rm "$container_id" > /dev/null 2>&1 || true
    echo "✅ Test container cleaned up"
}

# Verify gcloud authentication
verify_auth

# Authenticate gcloud and configure Docker auth
echo "🔐 Logging into GCP & configuring Docker auth..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet || handle_error "Failed to configure Docker authentication"
echo "✅ Done configuring Docker auth."

# Build Docker image
echo "🔨 Building serving container..."
docker build -f "${DOCKERFILE_PATH}/Dockerfile" -t "${IMAGE_URI}" . || handle_error "Failed to build Docker image"
echo "✅ Build complete."

# Test container locally (optional)
echo "🧪 Testing container locally..."
CONTAINER_ID=$(docker run -d -p 8080:8080 \
    -e MODEL_NAME=fraud_detection \
    -e LOG_LEVEL=INFO \
    -e BATCH_SIZE=32 \
    "${IMAGE_URI}") || handle_error "Failed to start test container"

test_container_health "$CONTAINER_ID"

# Push Docker image
echo "📤 Pushing image to Artifact Registry..."
docker push "${IMAGE_URI}" || handle_error "Failed to push Docker image"
echo "✅ Push complete."

echo "🎉 SUCCESS: Serving image ${IMAGE_URI} is now in Artifact Registry."
echo ""
echo "📋 Copy this image URI for deployment:"
echo "${IMAGE_URI}"