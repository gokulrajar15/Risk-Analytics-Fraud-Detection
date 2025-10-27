#!/usr/bin/env bash
set -euo pipefail

# === Variables – adjust as needed ===
PROJECT_ID="${PROJECT_ID:-your-gcp-project-id}"
REGION="${REGION:-us-central1}"
REPO_NAME="${REPO_NAME:-vertex-training-repo}"
IMAGE_NAME="${IMAGE_NAME:-iris-trainer}"
TAG="${TAG:-latest}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-.}"       # path where Dockerfile + context lives

# Full image URI for Artifact Registry
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}"

# === Script starts ===
echo "🔧 Building & Pushing Docker image to Artifact Registry"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Repository: $REPO_NAME"
echo "Image name: $IMAGE_NAME"
echo "Tag: $TAG"
echo "Image URI: $IMAGE_URI"
echo

# Authenticate gcloud (should have run once or ensure you have correct permissions)
echo "Logging into GCP & configuring Docker auth..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
echo "Done configuring Docker auth."

# Build Docker image
echo "Building Docker image..."
docker build -t "${IMAGE_URI}" "${DOCKERFILE_PATH}"
echo "Build complete."

# Push Docker image
echo "Pushing image to Artifact Registry..."
docker push "${IMAGE_URI}"
echo "Push complete."

echo "✅ Image ${IMAGE_URI} is now in Artifact Registry."
