$ErrorActionPreference = "Stop"

Write-Host "[1/2] Health check..." -ForegroundColor Cyan
$health = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8787/api/health"
$health | ConvertTo-Json -Depth 5

Write-Host "`n[2/2] Predict (skip_api=true)..." -ForegroundColor Cyan
$body = @{ skip_api = $true } | ConvertTo-Json
$pred = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8787/api/predict" -ContentType "application/json" -Body $body
$pred | ConvertTo-Json -Depth 6
