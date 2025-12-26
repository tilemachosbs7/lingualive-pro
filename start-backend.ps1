# LanguageTranslate Backend Starter
# Automatic backend server startup

Write-Host " LanguageTranslate Backend Starter" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
$venvPath = ".\backend\.venv\Scripts\Activate.ps1"
if (-Not (Test-Path $venvPath)) {
    Write-Host " Virtual environment not found!" -ForegroundColor Red
    Write-Host "Running: python -m venv backend\.venv" -ForegroundColor Yellow
    python -m venv backend\.venv
    Write-Host " Virtual environment created!" -ForegroundColor Green
}

# Activate virtual environment
Write-Host " Activating virtual environment..." -ForegroundColor Yellow
& $venvPath

# Check dependencies
Write-Host " Checking dependencies..." -ForegroundColor Yellow
$requirementsPath = ".\backend\requirements.txt"
if (Test-Path $requirementsPath) {
    pip install -q -r $requirementsPath
    Write-Host " Dependencies installed!" -ForegroundColor Green
} else {
    Write-Host "  requirements.txt not found" -ForegroundColor Yellow
}

# Check .env file
$envPath = ".\backend\.env"
if (-Not (Test-Path $envPath)) {
    Write-Host "  .env not found!" -ForegroundColor Yellow
    Write-Host "Creating .env from template..." -ForegroundColor Yellow
    if (Test-Path ".\backend\.env.example") {
        Copy-Item ".\backend\.env.example" -Destination $envPath -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host " All ready!" -ForegroundColor Green
Write-Host ""
Write-Host " Starting backend server..." -ForegroundColor Cyan
Write-Host " URL: http://localhost:8000" -ForegroundColor Cyan
Write-Host " Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host " Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start backend
Set-Location backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
