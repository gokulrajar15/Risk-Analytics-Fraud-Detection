# PowerShell version of build_and_push.sh
param(
    [string]$ProjectId = $env:PROJECT_ID,
    [string]$Region = $env:REGION,
    [string]$RepoName = $env:REPO_NAME,
    [string]$ImageName = $env:IMAGE_NAME,
    [string]$Tag = $env:TAG,
    [string]$DockerfilePath = "./training_job"
)

# Set default values if not provided
if (-not $ProjectId) { $ProjectId = "your-gcp-project-id" }
if (-not $Region) { $Region = "us-central1" }
if (-not $RepoName) { $RepoName = "vertex-training-repo" }
if (-not $ImageName) { $ImageName = "fraud-detection-trainer" }
if (-not $Tag) { $Tag = "latest" }

# Full image URI for Artifact Registry
$ImageUri = "$Region-docker.pkg.dev/$ProjectId/$RepoName/$ImageName" + ":" + $Tag

Write-Host "Building and Pushing Docker image to Artifact Registry" -ForegroundColor Cyan
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
    Write-Host '$env:PROJECT_ID="your-actual-project-id"' -ForegroundColor Yellow
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
    Write-Host "Building Docker image..." -ForegroundColor Yellow
    docker build -f "$DockerfilePath/Dockerfile" -t $ImageUri .
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build Docker image"
    }
    Write-Host "Build complete." -ForegroundColor Green

    # Push Docker image
    Write-Host "Pushing image to Artifact Registry..." -ForegroundColor Yellow
    docker push $ImageUri
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to push Docker image"
    }
    Write-Host "Push complete." -ForegroundColor Green

    Write-Host "SUCCESS: Image $ImageUri is now in Artifact Registry." -ForegroundColor Green
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    exit 1
}