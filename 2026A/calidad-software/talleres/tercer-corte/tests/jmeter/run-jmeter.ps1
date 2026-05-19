# Ejecuta el plan JMeter en modo no-GUI y genera reporte HTML.
# Requisito: Apache JMeter 5.x instalado y JMETER_HOME configurado (o jmeter en PATH).

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$JmxFile = Join-Path $ScriptDir "parking-api-load-test.jmx"
$ReportsDir = Join-Path $AppRoot "reports\jmeter"
$JtlFile = Join-Path $ReportsDir "results.jtl"
$HtmlReportDir = Join-Path $ReportsDir "html-report"

if (-not (Test-Path $ReportsDir)) {
    New-Item -ItemType Directory -Path $ReportsDir -Force | Out-Null
}

$jmeterCmd = Get-Command jmeter -ErrorAction SilentlyContinue
if (-not $jmeterCmd -and $env:JMETER_HOME) {
    $jmeterCmd = Get-Command (Join-Path $env:JMETER_HOME "bin\jmeter.bat") -ErrorAction SilentlyContinue
}
if (-not $jmeterCmd) {
    Write-Error "No se encontró JMeter. Instálalo y agrega 'jmeter' al PATH o define JMETER_HOME."
}

Write-Host "Comprobando backend en http://localhost:8080/api/user ..."
try {
    $null = Invoke-WebRequest -Uri "http://localhost:8080/api/user" -UseBasicParsing -TimeoutSec 5
    Write-Host "Backend OK."
} catch {
    Write-Warning "El backend no respondió. Levanta Docker: cd app/Docker && docker compose up -d"
}

Write-Host "Ejecutando prueba de carga..."
& $jmeterCmd.Source -n -t $JmxFile -l $JtlFile -e -o $HtmlReportDir

Write-Host ""
Write-Host "Listo."
Write-Host "  Resultados (.jtl): $JtlFile"
Write-Host "  Reporte HTML:      $HtmlReportDir\index.html"
