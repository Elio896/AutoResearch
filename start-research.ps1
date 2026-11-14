param([int]$Port = 8002)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$researchPython = $env:EVORESEARCH_PYTHON
if (-not $researchPython) {
    $researchPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $researchPython)) { $researchPython = 'python' }
}
& $researchPython -c "import fastapi,uvicorn,pandas,sklearn"
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Dependencies are missing. Use your existing Python environment, or follow START_HERE.md.'
    exit 1
}
Write-Host "Research page: http://127.0.0.1:$Port/research-agent"
Write-Host 'Stop with Ctrl+C. API keys are read from server environment variables only.'
& $researchPython -m uvicorn api:create_app --factory --host 127.0.0.1 --port $Port
