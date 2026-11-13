param(
    [string]$Csv = 'bench\tasks\t3_clean\data.csv',
    [string]$Target = 'target',
    [string]$JevModel = 'jev-latest',
    [string]$CloudModel = 'deepseek-flash'
)
$ErrorActionPreference = 'Stop'
$previousKeys = @{}
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    foreach ($keyName in @('TYPESAFE_API_KEY', 'DEEPSEEK_API_KEY')) {
        $previousKeys[$keyName] = [Environment]::GetEnvironmentVariable($keyName, 'Process')
        $secureKey = Read-Host "$keyName (hidden; this run only)" -AsSecureString
        $pointer = [IntPtr]::Zero
        try {
            $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
            [Environment]::SetEnvironmentVariable($keyName, [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer), 'Process')
        } finally {
            if ($pointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
            $secureKey.Dispose()
        }
    }
    $resultFile = 'artifacts\jev_' + [guid]::NewGuid().ToString('N') + '.json'
    & .\.venv\Scripts\python.exe orchestrator.py $Csv --target $Target --enable-cloud --enable-jev --cloud-model $CloudModel --jev-model $JevModel --max-cloud-calls 1 --max-jev-calls 1 --max-experiments 3 --max-calls 3 --max-seconds 120 --output $resultFile
    if ($LASTEXITCODE -ne 0) { throw 'Research failed; inspect the non-secret error above.' }
} finally {
    foreach ($keyName in $previousKeys.Keys) {
        [Environment]::SetEnvironmentVariable($keyName, $previousKeys[$keyName], 'Process')
    }
    Pop-Location
}
