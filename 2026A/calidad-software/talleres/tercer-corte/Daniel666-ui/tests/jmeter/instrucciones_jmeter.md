# Instrucciones JMeter (CLI) — Task Manager

## Requisitos
- Java 11+ instalado en tu equipo (para ejecutar JMeter).
- Apache JMeter 5.x descargado y descomprimido.
- Aplicación Task Manager ejecutándose en `http://localhost:8080`.

## Archivo del plan
- `tests/jmeter/plan_pruebas_taskmanager.jmx`

## Ejecutar desde CLI (modo no-GUI)
Desde la raíz del repositorio:

### Windows (PowerShell)
```powershell
$JMETER_HOME="C:\tools\apache-jmeter-5.6.3"
& "$JMETER_HOME\bin\jmeter.bat" -n `
  -t "tests\jmeter\plan_pruebas_taskmanager.jmx" `
  -l "reports\entrega-2\jmeter\resultados.jtl" `
  -e -o "reports\entrega-2\jmeter\html"
```

### Linux/Mac (bash)
```bash
JMETER_HOME=/opt/apache-jmeter-5.6.3
"$JMETER_HOME/bin/jmeter" -n \
  -t tests/jmeter/plan_pruebas_taskmanager.jmx \
  -l reports/entrega-2/jmeter/resultados.jtl \
  -e -o reports/entrega-2/jmeter/html
```

## Parametrizar host/puerto
El plan define variables `BASE_HOST` y `BASE_PORT`. Puedes sobrescribirlas en CLI:

```bash
"$JMETER_HOME/bin/jmeter" -n \
  -t tests/jmeter/plan_pruebas_taskmanager.jmx \
  -JBASE_HOST=localhost \
  -JBASE_PORT=8080
```

## Dónde ver resultados
- Archivo JTL: `reports/entrega-2/jmeter/resultados.jtl`
- Reporte HTML: `reports/entrega-2/jmeter/html/index.html`
