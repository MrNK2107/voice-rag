# Reproducible offline Health-track demo (no API key, no Qdrant server, no model download).
#
#   .\scripts\demo.ps1                  # console walkthrough + self-check
#   .\scripts\demo.ps1 -Serve [-Port N] # same, then serve the web UI in demo mode
param([switch]$Serve, [int]$Port = 8000)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

& $py scripts/demo.py --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Serve) {
    Write-Host ""
    Write-Host "Serving the demo UI at http://localhost:$Port  (Ctrl+C to stop)"
    $env:VOICE_RAG_DEMO = "1"
    & $py -m uvicorn app.main:app --port $Port
}
