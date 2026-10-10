$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'C:\Users\MARS\Desktop\College\Minor Project\Project Development'
try {
    & '.\.venv\Scripts\python.exe' -u -B -m research.short_screening.s97_final_convnext_warmstart --resume
    $s97WorkerExitCode = $LASTEXITCODE
    [pscustomobject]@{ FinishedAt=(Get-Date).ToString('o'); ExitCode=$s97WorkerExitCode; Mode='unchanged_checkpoint_resume' } | ConvertTo-Json | Set-Content -LiteralPath 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\short_screening\final_convnext_warmstart_v1\recovery_20261010T092614Z\exit_status.json' -Encoding utf8
    exit $s97WorkerExitCode
} catch {
    [pscustomobject]@{ FinishedAt=(Get-Date).ToString('o'); Error=$_.Exception.Message; Mode='unchanged_checkpoint_resume' } | ConvertTo-Json | Set-Content -LiteralPath 'C:\Users\MARS\Desktop\College\Minor Project\Project Development\results\short_screening\final_convnext_warmstart_v1\recovery_20261010T092614Z\exit_status.json' -Encoding utf8
    throw
}
