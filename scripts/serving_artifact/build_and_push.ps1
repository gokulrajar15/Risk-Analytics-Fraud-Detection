# Build and push serving container to Artifact Registry
# PowerShell script

param(
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    [string]$Region = $env:GCP_REGION,
    [string]$Repository = "serving-repo",
    [string]$ImageName = "fraud-detection-serving",
    [string]$Tag = "latest"
)

# Check if required parameters are provided
if (-not $ProjectId) {
    Write-Error "PROJECT_ID is required. Set GCP_PROJECT_ID environment variable or pass -ProjectId parameter."
    exit 1
}

if (-not $Region) {
    Write-Error "REGION is required. Set GCP_REGION environment variable or pass -Region parameter."
    exit 1
}

$FullImageName = "${Region}-docker.pkg.dev/${ProjectId}/${Repository}/${ImageName}:${Tag}"

Write-Host "Building and pushing serving container..." -ForegroundColor Green
Write-Host "Project ID: $ProjectId" -ForegroundColor Yellow
Write-Host "Region: $Region" -ForegroundColor Yellow
Write-Host "Image: $FullImageName" -ForegroundColor Yellow

# Configure Docker for Artifact Registry
Write-Host "Configuring Docker authentication..." -ForegroundColor Cyan
gcloud auth configure-docker "${Region}-docker.pkg.dev" --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to configure Docker authentication"
    exit 1
}

# Create repository if it doesn't exist
Write-Host "Creating Artifact Registry repository (if not exists)..." -ForegroundColor Cyan
gcloud artifacts repositories create $Repository `
    --repository-format=docker `
    --location=$Region `
    --description="Serving container repository" `
    --quiet 2>$null

# Build the Docker image
Write-Host "Building Docker image..." -ForegroundColor Cyan
docker build -t $FullImageName .

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to build Docker image"
    exit 1
}

# Push the Docker image
Write-Host "Pushing Docker image to Artifact Registry..." -ForegroundColor Cyan
docker push $FullImageName

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to push Docker image"
    exit 1
}

Write-Host "Successfully built and pushed serving container!" -ForegroundColor Green
Write-Host "Image URI: $FullImageName" -ForegroundColor Yellow

# Update .env file with the new image URI
$EnvFile = "..\.env"
if (Test-Path $EnvFile) {
    Write-Host "Updating .env file with new serving image URI..." -ForegroundColor Cyan
    $content = Get-Content $EnvFile
    $updatedContent = $content -replace '^SERVING_IMAGE\s*=.*', "SERVING_IMAGE=`"$FullImageName`""
    
    if ($content -notmatch '^SERVING_IMAGE\s*=') {
        $updatedContent += "SERVING_IMAGE=`"$FullImageName`""
    }
    
    $updatedContent | Set-Content $EnvFile
    Write-Host "Updated .env file with serving image URI" -ForegroundColor Green
}