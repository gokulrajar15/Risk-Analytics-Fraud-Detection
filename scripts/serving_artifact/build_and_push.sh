#!/bin/bash

# Build and push serving container to Artifact Registry
# Bash script

# Default values from environment variables
PROJECT_ID=${GCP_PROJECT_ID}
REGION=${GCP_REGION}
REPOSITORY="serving-repo"
IMAGE_NAME="fraud-detection-serving"
TAG="latest"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--project)
            PROJECT_ID="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        --repository)
            REPOSITORY="$2"
            shift 2
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -p, --project     GCP Project ID (default: \$GCP_PROJECT_ID)"
            echo "  -r, --region      GCP Region (default: \$GCP_REGION)"
            echo "  --repository      Artifact Registry repository (default: serving-repo)"
            echo "  --image-name      Docker image name (default: fraud-detection-serving)"
            echo "  --tag             Image tag (default: latest)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required parameters
if [[ -z "$PROJECT_ID" ]]; then
    echo "Error: PROJECT_ID is required. Set GCP_PROJECT_ID environment variable or pass -p parameter."
    exit 1
fi

if [[ -z "$REGION" ]]; then
    echo "Error: REGION is required. Set GCP_REGION environment variable or pass -r parameter."
    exit 1
fi

FULL_IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"

echo "Building and pushing serving container..."
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Image: $FULL_IMAGE_NAME"

# Configure Docker for Artifact Registry
echo "Configuring Docker authentication..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

if [[ $? -ne 0 ]]; then
    echo "Error: Failed to configure Docker authentication"
    exit 1
fi

# Create repository if it doesn't exist
echo "Creating Artifact Registry repository (if not exists)..."
gcloud artifacts repositories create "$REPOSITORY" \
    --repository-format=docker \
    --location="$REGION" \
    --description="Serving container repository" \
    --quiet 2>/dev/null

# Build the Docker image
echo "Building Docker image..."
docker build -t "$FULL_IMAGE_NAME" .

if [[ $? -ne 0 ]]; then
    echo "Error: Failed to build Docker image"
    exit 1
fi

# Push the Docker image
echo "Pushing Docker image to Artifact Registry..."
docker push "$FULL_IMAGE_NAME"

if [[ $? -ne 0 ]]; then
    echo "Error: Failed to push Docker image"
    exit 1
fi

echo "Successfully built and pushed serving container!"
echo "Image URI: $FULL_IMAGE_NAME"

# Update .env file with the new image URI
ENV_FILE="../.env"
if [[ -f "$ENV_FILE" ]]; then
    echo "Updating .env file with new serving image URI..."
    
    # Create backup
    cp "$ENV_FILE" "${ENV_FILE}.backup"
    
    # Update or add SERVING_IMAGE line
    if grep -q "^SERVING_IMAGE" "$ENV_FILE"; then
        sed -i "s|^SERVING_IMAGE=.*|SERVING_IMAGE=\"$FULL_IMAGE_NAME\"|" "$ENV_FILE"
    else
        echo "SERVING_IMAGE=\"$FULL_IMAGE_NAME\"" >> "$ENV_FILE"
    fi
    
    echo "Updated .env file with serving image URI"
fi