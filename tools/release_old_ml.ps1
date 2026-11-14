param([int]$Port=8012)
$ErrorActionPreference = 'Stop'
$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
foreach ($listener in $listeners) {
    $taskId = $listener.OwningProcess
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$taskId"
    $line = $proc.CommandLine
    $known = ($proc.Name -match '^python(w)?\.exe$') -and (($line -match 'uvicorn' -and $line -match '(ml_api|api):create_app' -and $line -match [regex]::Escape([string]$Port)) -or ($line -match 'tools[\/]start_connected_ml\.py'))
    if (-not $known) { exit 2 }
    Stop-Process -Id $taskId -ErrorAction Stop
}
exit 0
