$root = $PSScriptRoot
$python = "$root\.venv\Scripts\python.exe"
$env = "PYTHONIOENCODING=utf-8"

function Start-Service($title, $module) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:PYTHONIOENCODING='utf-8'; Set-Location '$root'; & '$python' -m $module" -WindowStyle Normal
}

Write-Host "Starting all Legal Multi-Agent services..." -ForegroundColor Cyan

Start-Service "Registry"         "registry"
Write-Host "Waiting for Registry to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Start-Service "Tax Agent"        "tax_agent"
Start-Service "Compliance Agent" "compliance_agent"
Start-Sleep -Seconds 4

Start-Service "Law Agent"        "law_agent"
Start-Sleep -Seconds 3

Start-Service "Customer Agent"   "customer_agent"
Start-Sleep -Seconds 2

Start-Service "UI"               "ui.app"

Write-Host ""
Write-Host "All services started!" -ForegroundColor Green
Write-Host "Web UI: http://localhost:8080" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit this window..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
