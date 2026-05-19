# Pruebas de rendimiento — JMeter

Plan de carga para el backend Spring Boot (`http://localhost:8080`).

## Requisitos

1. **Docker Desktop** con el stack levantado:
   ```bash
   cd app/Docker
   docker compose up -d
   ```
2. **Apache JMeter 5.x** ([descarga](https://jmeter.apache.org/download_jmeter.cgi))
3. Verificar que la API responde (BD vacía devuelve `[]`):
   ```bash
   curl http://localhost:8080/api/user
   ```

## Archivo del plan

| Archivo | Descripción |
|---------|-------------|
| `parking-api-load-test.jmx` | Plan con 3 escenarios en serie |

### Escenarios

| Grupo | Usuarios | Ramp-up | Iteraciones | Endpoints |
|-------|----------|---------|-------------|-----------|
| 01 - Normal | 5 | 10 s | 10 | GET `/api/user`, `/api/tarifas`, `/api/reservas` |
| 02 - Carga | 25 | 30 s | 5 | mismos |
| 03 - Estrés | 50 | 20 s | 3 | mismos |

No se requieren usuarios en la BD: los GET devuelven listas vacías con HTTP 200.

Variables del plan: `HOST=localhost`, `PORT=8080` (editables en JMeter GUI).

## Ejecución

### Opción A — Interfaz gráfica

1. Abrir JMeter → **File → Open** → `parking-api-load-test.jmx`
2. Pulsar **Start** (triángulo verde)
3. Revisar **Summary Report** y **Aggregate Report**

### Opción B — Línea de comandos (reporte HTML)

Desde PowerShell, en esta carpeta:

```powershell
.\run-jmeter.ps1
```

Salida:

- `app/reports/jmeter/results.jtl`
- `app/reports/jmeter/html-report/index.html`

## Métricas a documentar (entrega)

Del **Aggregate Report** o del HTML:

- Tiempo medio de respuesta
- Percentil 90 / 95
- Throughput (peticiones/s)
- % de errores

## Siguiente paso

Cuando tengas datos en la BD (tras usar el front), se puede añadir un **Setup Thread Group** con `POST /api/user` para pruebas con escritura.
