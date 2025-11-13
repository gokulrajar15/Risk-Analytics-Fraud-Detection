# PowerShell version of build_and_push.sh for serving container
param(
    [string]$ProjectId = $env:PROJECT_ID,
    [string]$Region = $env:REGION,
    [string]$RepoName = $env:SERVING_REPO_NAME,
    [string]$ImageName = $env:SERVING_IMAGE_NAME,
    [string]$Tag = $env:TAG,
    [string]$DockerfilePath = "./serving_container"
)

# Set default values if not provided
if (-not $ProjectId) { $ProjectId = "your-gcp-project-id" }
if (-not $Region) { $Region = "us-central1" }
if (-not $RepoName) { $RepoName = "vertex-ai-customservingartifact" }
if (-not $ImageName) { $ImageName = "fraud-detection-serving" }
if (-not $Tag) { $Tag = "latest" }

# Full image URI for Artifact Registry
$ImageUri = "$Region-docker.pkg.dev/$ProjectId/$RepoName/$ImageName" + ":" + $Tag

Write-Host "Building and Pushing Serving Container to Artifact Registry" -ForegroundColor Cyan
Write-Host "Project: $ProjectId" -ForegroundColor Green
Write-Host "Region: $Region" -ForegroundColor Green
Write-Host "Repository: $RepoName" -ForegroundColor Green
Write-Host "Image name: $ImageName" -ForegroundColor Green
Write-Host "Tag: $Tag" -ForegroundColor Green
Write-Host "Image URI: $ImageUri" -ForegroundColor Green
Write-Host ""

# Check if required variables are set
if ($ProjectId -eq "your-gcp-project-id") {
    Write-Host "Please set your actual GCP Project ID!" -ForegroundColor Red
    Write-Host "Use: " -NoNewline
    Write-Host '$env:PROJECT_ID="poetic-velocity-459409-f2"' -ForegroundColor Yellow
    exit 1
}

try {
    # Authenticate gcloud and configure Docker auth
    Write-Host "Logging into GCP and configuring Docker auth..." -ForegroundColor Yellow
    gcloud auth configure-docker "$Region-docker.pkg.dev" --quiet
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to configure Docker authentication"
    }
    Write-Host "Done configuring Docker auth." -ForegroundColor Green

    # Build Docker image
    Write-Host "Building serving container..." -ForegroundColor Yellow
    docker build -f "$DockerfilePath/Dockerfile" -t $ImageUri .
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build Docker image"
    }
    Write-Host "Build complete." -ForegroundColor Green

    # Test container locally (optional health check)
    Write-Host "Testing container locally..." -ForegroundColor Yellow
    $ContainerId = docker run -d -p 8080:8080 `
        -e MODEL_NAME=fraud_detection `
        -e LOG_LEVEL=INFO `
        -e BATCH_SIZE=32 `
        $ImageUri
    
    if ($LASTEXITCODE -eq 0) {
        Start-Sleep -Seconds 10
        Write-Host "Container started with ID: $ContainerId" -ForegroundColor Green
        
        # Test health endpoint
        try {
            $Response = Invoke-RestMethod -Uri "http://localhost:8080/health" -Method Get -TimeoutSec 15
            Write-Host "Health check passed: $($Response | ConvertTo-Json -Compress)" -ForegroundColor Green
        }
        catch {
            Write-Host "Health check warning (container may still be starting): $_" -ForegroundColor Yellow
        }
        
        # Stop test container
        docker stop $ContainerId | Out-Null
        docker rm $ContainerId | Out-Null
        Write-Host "Test container stopped and removed." -ForegroundColor Green
    }

    # Push Docker image
    Write-Host "Pushing image to Artifact Registry..." -ForegroundColor Yellow
    docker push $ImageUri
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to push Docker image"
    }
    Write-Host "Push complete." -ForegroundColor Green

    Write-Host "SUCCESS: Serving image $ImageUri is now in Artifact Registry." -ForegroundColor Green
    
    # Output the image URI for easy copying
    Write-Host ""
    Write-Host "Copy this image URI for deployment:" -ForegroundColor Yellow
    Write-Host $ImageUri -ForegroundColor White
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    exit 1
}