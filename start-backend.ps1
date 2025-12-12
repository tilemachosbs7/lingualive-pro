# LanguageTranslate Backend Starter
# Αυτόματη εκκίνηση του backend server

Write-Host "🚀 LanguageTranslate Backend Starter" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Έλεγχος αν υπάρχει το virtual environment
$venvPath = ".\.venv\Scripts\Activate.ps1"
if (-Not (Test-Path $venvPath)) {
    Write-Host "❌ Το virtual environment δεν βρέθηκε!" -ForegroundColor Red
    Write-Host "Τρέχουμε: python -m venv .venv" -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "✅ Virtual environment δημιουργήθηκε!" -ForegroundColor Green
}

# Ενεργοποίηση virtual environment
Write-Host "🔧 Ενεργοποίηση virtual environment..." -ForegroundColor Yellow
& $venvPath

# Έλεγχος dependencies
Write-Host "📦 Έλεγχος dependencies..." -ForegroundColor Yellow
$requirementsPath = ".\backend\requirements.txt"
if (Test-Path $requirementsPath) {
    pip install -q -r $requirementsPath
    Write-Host "✅ Dependencies εγκατεστημένα!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Το requirements.txt δεν βρέθηκε" -ForegroundColor Yellow
}

# Έλεγχος .env file
$envPath = ".\backend\.env"
if (-Not (Test-Path $envPath)) {
    Write-Host "⚠️  Το .env δεν βρέθηκε!" -ForegroundColor Yellow
    Write-Host "Δημιουργία .env από template..." -ForegroundColor Yellow
    Copy-Item ".\backend\.env.example" -Destination $envPath -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "✅ Όλα έτοιμα!" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Εκκίνηση backend server..." -ForegroundColor Cyan
Write-Host "📍 URL: http://localhost:8000" -ForegroundColor Cyan
Write-Host "📚 Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 Πάτα Ctrl+C για να σταματήσεις το server" -ForegroundColor Yellow
Write-Host ""

# Εκκίνηση backend
Set-Location backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
