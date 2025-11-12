#!/usr/bin/env bash
set -euo pipefail

# === Variables – adjust as needed ===
PROJECT_ID="${PROJECT_ID:-your-gcp-project-id}"
REGION="${REGION:-us-central1}"
REPO_NAME="${REPO_NAME:-vertex-training-repo}"
IMAGE_NAME="${IMAGE_NAME:-fraud-detection-trainer}"
TAG="${TAG:-latest}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-./training_job}"

# Full image URI for Artifact Registry
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}"

# === Script starts ===
echo "🔧 Building & Pushing Docker image to Artifact Registry"
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
    echo "Use: export PROJECT_ID=\"your-actual-project-id\""
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

# Verify gcloud authentication
verify_auth

# Authenticate gcloud and configure Docker auth
echo "🔐 Logging into GCP & configuring Docker auth..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet || handle_error "Failed to configure Docker authentication"
echo "✅ Done configuring Docker auth."

# Build Docker image
echo "🔨 Building Docker image..."
docker build -f "${DOCKERFILE_PATH}/Dockerfile" -t "${IMAGE_URI}" . || handle_error "Failed to build Docker image"
echo "✅ Build complete."

# Push Docker image
echo "📤 Pushing image to Artifact Registry..."
docker push "${IMAGE_URI}" || handle_error "Failed to push Docker image"
echo "✅ Push complete."

echo "🎉 SUCCESS: Image ${IMAGE_URI} is now in Artifact Registry."
