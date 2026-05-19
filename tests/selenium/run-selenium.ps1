# Pruebas Selenium con Brave. Requiere Docker (:8080) y ng serve (:4200).

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Test-Endpoint($Url, $Label) {
    try {
        $null = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 8
        Write-Host "$Label OK ($Url)"
        return $true
    } catch {
        Write-Warning "$Label no disponible ($Url)"
        return $false
    }
}

$bravePaths = @(
    "${env:ProgramFiles}\BraveSoftware\Brave-Browser\Application\brave.exe",
    "${env:ProgramFiles(x86)}\BraveSoftware\Brave-Browser\Application\brave.exe",
    "${env:LOCALAPPDATA}\BraveSoftware\Brave-Browser\Application\brave.exe"
)
$bravePath = $bravePaths | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $bravePath) {
    Write-Error "Brave no está instalado. Descárgalo en https://brave.com/download/"
}

Write-Host "Brave: $bravePath"

$null = Test-Endpoint "http://localhost:8080/api/user" "Backend"
$null = Test-Endpoint "http://localhost:4200/home" "Frontend"

Push-Location $ScriptDir
try {
    Write-Host "`n=== Ejecutando tests en Brave ===" -ForegroundColor Cyan
    mvn test -Dbrowser=brave
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "`nReportes: app/reports/selenium/surefire-reports" -ForegroundColor Green
} finally {
    Pop-Location
}
