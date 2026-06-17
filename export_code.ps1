Write-Host "Starting EdgeCraft AI export process..." -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green
Write-Host ""

# ----------------------------------------
# 1. Process Frontend
# ----------------------------------------
$frontendPath = ""
if (Test-Path "frontend/src") {
    $frontendPath = "frontend/src"
} elseif (Test-Path "frontend") {
    $frontendPath = "frontend"
}

if ($frontendPath -ne "") {
    Write-Host "[!] Zipping $frontendPath into frontend.zip..." -ForegroundColor Yellow
    Compress-Archive -Path $frontendPath -DestinationPath "frontend.zip" -Force
    
    Write-Host "[!] Converting frontend.zip to frontend.md using markitdown..." -ForegroundColor Yellow
    markitdown frontend.zip > frontend.md
    
    Write-Host "[SUCCESS] Frontend processed successfully!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "[ERROR] Directory frontend/ or frontend/src/ not found! Skipping frontend..." -ForegroundColor Red
    Write-Host ""
}

# ----------------------------------------
# 2. Process Backend
# ----------------------------------------
if (Test-Path "backend/app") {
    Write-Host "[!] Zipping backend/app/ into backend.zip..." -ForegroundColor Yellow
    Compress-Archive -Path "backend/app" -DestinationPath "backend.zip" -Force
    
    Write-Host "[!] Converting backend.zip to backend.md using markitdown..." -ForegroundColor Yellow
    markitdown backend.zip > backend.md
    
    Write-Host "[SUCCESS] Backend processed successfully!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "[ERROR] Directory backend/app/ not found! Skipping backend..." -ForegroundColor Red
    Write-Host ""
}

Write-Host "=======================================" -ForegroundColor Green
Write-Host "All exports complete! Check your directory for the new .md files." -ForegroundColor Green