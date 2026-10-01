$ErrorActionPreference = "Stop"

# ============================================================
# Adaptive-MMRAG Startup Script
# Auto-detects Python and Ollama on the current machine.
# ============================================================

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " Adaptive-MMRAG Startup" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# ------ Force HF offline mode (prevents huggingface network calls) ------
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUTF8 = "1"

# ------ Tesseract PATH (adjust if your installation is elsewhere) ------
$tesseractPath = "C:\Program Files\Tesseract-OCR"
if (Test-Path $tesseractPath) {
    $env:PATH = "$tesseractPath;" + $env:PATH
    Write-Host "[OK] Tesseract found at $tesseractPath" -ForegroundColor Green
} else {
    Write-Host "[WARN] Tesseract-OCR not found at $tesseractPath — OCR will be skipped" -ForegroundColor Yellow
}

# ------ Locate Ollama executable (check common locations) ------
$ollamaCandidates = @(
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe",
    "C:\Users\GADDAM ADITHYA\AppData\Local\Programs\Ollama\ollama.exe",
    "C:\Program Files\Ollama\ollama.exe",
    "ollama"  # If already on PATH
)

$ollamaExe = $null
foreach ($candidate in $ollamaCandidates) {
    try {
        if ($candidate -eq "ollama") {
            $found = Get-Command "ollama" -ErrorAction SilentlyContinue
            if ($found) { $ollamaExe = "ollama"; break }
        } elseif (Test-Path $candidate) {
            $ollamaExe = $candidate; break
        }
    } catch {}
}

if ($ollamaExe) {
    Write-Host "[OK] Ollama found: $ollamaExe" -ForegroundColor Green
    # Start Ollama server in background (silently, OK if already running)
    try {
        Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        Write-Host "[OK] Ollama server started (or already running)" -ForegroundColor Green
    } catch {
        Write-Host "[WARN] Could not start Ollama automatically: $_" -ForegroundColor Yellow
        Write-Host "       Please run 'ollama serve' manually in another terminal." -ForegroundColor Yellow
    }
} else {
    Write-Host "[WARN] Ollama executable not found — general chat will be unavailable." -ForegroundColor Yellow
    Write-Host "       Download from: https://ollama.com/download" -ForegroundColor Yellow
    Write-Host "       After install, run: ollama pull qwen2.5:3b" -ForegroundColor Yellow
}

# ------ Locate Python ------
$pythonCandidates = @(
    ".\.venv\Scripts\python.exe",
    ".\venv\Scripts\python.exe",
    "python"
)

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    try {
        if ($candidate -eq "python") {
            $found = Get-Command "python" -ErrorAction SilentlyContinue
            if ($found) { $pythonExe = "python"; break }
        } elseif (Test-Path $candidate) {
            # Validate it actually runs (venv python.exe can be a broken stub)
            $ver = & $candidate --version 2>&1
            if ($LASTEXITCODE -eq 0) { $pythonExe = $candidate; break }
        }
    } catch {}
}

if (-not $pythonExe) {
    Write-Host "[ERROR] No working Python found. Install Python 3.10+ and run:" -ForegroundColor Red
    Write-Host "        python -m pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Python: $pythonExe" -ForegroundColor Green

# ------ Install ollama package if missing ------
$ollamaCheck = & $pythonExe -c "import ollama" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[INFO] Installing ollama Python package..." -ForegroundColor Cyan
    & $pythonExe -m pip install ollama --quiet
}

Write-Host ""
Write-Host "[START] Launching Flask server..." -ForegroundColor Cyan
Write-Host "        Access at: http://localhost:5000" -ForegroundColor White
Write-Host ""

& $pythonExe "app1.py"
