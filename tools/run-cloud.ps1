param(
    [string]$Csv = 'data\iris.csv',
    [string]$Target = 'target',
    [string]$Model = 'deepseek-flash'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$previousKey = $env:DEEPSEEK_API_KEY
$keyPointer = [IntPtr]::Zero
Push-Location -LiteralPath $projectRoot
try {
    $secureKey = Read-Host 'DeepSeek API key (hidden; this run only)' -AsSecureString
    $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    $resultFile = 'artifacts\cloud_' + [guid]::NewGuid().ToString('N') + '.json'
    & .\.venv\Scripts\python.exe orchestrator.py $Csv --target $Target --enable-cloud --cloud-model $Model --max-cloud-calls 1 --max-experiments 3 --max-calls 3 --max-seconds 120 --output $resultFile
    if ($LASTEXITCODE -ne 0) { throw 'Research command failed; check the non-secret error above.' }
} finally {
    $env:DEEPSEEK_API_KEY = $previousKey
    if ($keyPointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer) }
    if ($secureKey) { $secureKey.Dispose() }
    Pop-Location
}
