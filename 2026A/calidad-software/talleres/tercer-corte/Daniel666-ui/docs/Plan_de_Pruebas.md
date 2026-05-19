# Plan de Pruebas — Task Manager
**Materia:** Calidad de Software 2026A  
**Proyecto:** Task Manager  
**Versión:** 1.0.0

## 1. Objetivo
Definir la estrategia, tipos de pruebas, herramientas y procedimientos para validar funcionalidad, seguridad, rendimiento y calidad de código del sistema.

## 2. Alcance de pruebas
Incluye:
- Pruebas unitarias e integración (JUnit 5, MockMvc).
- Cobertura (JaCoCo).
- Análisis estático (SonarQube).
- Pruebas funcionales UI (Selenium, Page Object Model, ExtentReports).
- Pruebas de rendimiento (JMeter) sobre la API.

## 3. Entorno
- Sistema operativo: Windows (Docker Desktop).
- Servicios (Docker Compose):
  - App: `http://localhost:8080`
  - SonarQube: `http://localhost:9000`
  - Selenium (standalone recomendado para ejecución local): `http://localhost:4444`

Credenciales:
- ADMIN: `admin/admin`
- USER: `user/user`

## 4. Herramientas
- **JUnit 5 + Spring Boot Test**: pruebas unitarias e integración.
- **JaCoCo**: cobertura de pruebas y reporte HTML/XML.
- **SonarQube**: análisis estático y Quality Gate.
- **Selenium WebDriver 4**: pruebas UI con Page Object Model.
- **ExtentReports**: reporte HTML de Selenium.
- **JMeter (Docker)**: plan de pruebas `.jmx` y reporte HTML.

## 5. Estrategia de pruebas
### 5.1 Unitarias/Integración
- Servicios y validaciones (dominio).
- Controladores REST con MockMvc.
- Reglas de seguridad y accesos.

### 5.2 UI (Selenium)
- Flujo de login.
- CRUD de tareas por UI.
- Filtro por estado.
- Verificación de elementos estables (`id` y `data-testid`).

### 5.3 Rendimiento (JMeter)
Escenarios definidos en `tests/jmeter/plan_pruebas_taskmanager.jmx`:
- Normal
- Carga
- Estrés (para ejecución prolongada; opcional)

## 6. Casos de prueba UI (mínimo 6)
- **TC01** Login exitoso con `admin/admin`.
- **TC02** Login fallido y validación de mensaje.
- **TC03** Crear tarea y validar que aparece en lista.
- **TC04** Editar tarea y validar cambios.
- **TC05** Eliminar tarea y validar que desaparece.
- **TC06** Filtrar por estado COMPLETED y validar filas.

## 7. Criterios
### 7.1 Entrada
- Servicios levantados con Docker Compose.
- Usuarios semilla creados (`admin`, `user`).
- Para Selenium: driver remoto disponible (Grid o standalone).

### 7.2 Salida
- Pruebas unitarias/integración en verde (build OK).
- Reporte JaCoCo generado en `reports/entrega-2/jacoco/`.
- SonarQube con análisis cargado y Quality Gate evaluado.
- Reporte JMeter HTML generado.
- Reporte Selenium (Extent) generado.

## 8. Procedimientos de ejecución (comandos)
### 8.1 Levantar servicios base
```powershell
docker compose up -d --build
```

### 8.2 Pruebas JUnit + JaCoCo
```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace/app maven:3.9-eclipse-temurin-11 mvn clean verify
```
Evidencias:
- `reports/entrega-2/jacoco/jacoco.xml`
- `reports/entrega-2/jacoco/index.html`

### 8.3 SonarQube (con token en variable de entorno)
```powershell
$env:SONAR_TOKEN="TU_TOKEN"
docker run --rm --network taskmanager-network -v "${PWD}:/workspace" -w /workspace/app maven:3.9-eclipse-temurin-11 mvn verify sonar:sonar "-Dsonar.host.url=http://sonarqube:9000" "-Dsonar.projectKey=taskmanager" "-Dsonar.login=$env:SONAR_TOKEN"
```

### 8.4 JMeter (Docker) contra el servicio `app`
```powershell
Remove-Item -Force ".\reports\entrega-2\jmeter\results.jtl" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force ".\reports\entrega-2\jmeter\html" -ErrorAction SilentlyContinue
docker run --rm --network taskmanager-network -v "${PWD}:/workspace" -w /workspace justb4/jmeter:5.5 -n -t /workspace/tests/jmeter/plan_pruebas_taskmanager.jmx -JBASE_HOST=app -JBASE_PORT=8080 -l /workspace/reports/entrega-2/jmeter/results.jtl -e -o /workspace/reports/entrega-2/jmeter/html
```
Evidencia:
- `reports/entrega-2/jmeter/html/index.html`

### 8.5 Selenium (standalone-chrome)
1) Descargar imagen (una sola vez):
```powershell
docker pull selenium/standalone-chrome:4.21.0
```
2) Levantar contenedor:
```powershell
docker rm -f tm-selenium-standalone 2>$null
docker run -d --name tm-selenium-standalone --network taskmanager-network -p 4444:4444 --shm-size 2g selenium/standalone-chrome:4.21.0
```
3) Ejecutar pruebas:
```powershell
docker run --rm --network taskmanager-network -v "${PWD}:/workspace" -w /workspace maven:3.9-eclipse-temurin-11 mvn -f tests/selenium/pom.xml test -DremoteUrl=http://tm-selenium-standalone:4444/wd/hub -Dbrowser=chrome -Dheadless=true -DbaseUrl=http://app:8080
```
Evidencia:
- `reports/entrega-2/selenium/extent-report.html`

## 9. Riesgos y mitigaciones
- **Descarga de imágenes Selenium falla por red (EOF):** cambiar temporalmente de red (hotspot) para completar el pull.
- **Intermitencias UI en contenedor:** usar esperas explícitas y validación de URL/DOM antes de interactuar.

## 10. Alineación con ISO/IEC 29119
Este documento sigue la intención de ISO/IEC 29119 (familia de estándares de pruebas):
- **ISO/IEC 29119-1 (Conceptos y definiciones):** tipos de prueba, criterios y entorno (secciones 2–4, 7).
- **ISO/IEC 29119-2 (Procesos de prueba):** estrategia, entradas/salidas, criterios de entrada/salida y riesgos (secciones 5, 7, 9).
- **ISO/IEC 29119-3 (Documentación de prueba):** casos de prueba y trazabilidad (secciones 11–12).

## 11. Especificación de casos de prueba (ISO/IEC 29119-3)
Formato: ID, objetivo, precondiciones, pasos, resultado esperado, postcondiciones.

### TC01 — Login exitoso (admin)
- **Objetivo:** validar autenticación correcta y redirección a dashboard.
- **Precondiciones:** App levantada; usuario `admin/admin` existe.
- **Pasos:** (1) Abrir `/login` (2) Ingresar `admin/admin` (3) Enviar formulario.
- **Resultado esperado:** navegación a `/dashboard` y render de elementos del dashboard.
- **Postcondiciones:** sesión autenticada activa.

### TC02 — Login fallido
- **Objetivo:** validar manejo de credenciales inválidas.
- **Precondiciones:** App levantada.
- **Pasos:** (1) Abrir `/login` (2) Ingresar `admin/wrong` (3) Enviar formulario.
- **Resultado esperado:** permanecer en login y mostrar mensaje “Credenciales inválidas”.
- **Postcondiciones:** no se crea sesión autenticada.

### TC03 — Crear tarea (CRUD)
- **Objetivo:** validar creación de tarea por UI.
- **Precondiciones:** usuario autenticado (admin/admin).
- **Pasos:** (1) Ir a `/tasks` (2) Click “New Task” (3) Completar campos válidos (4) Guardar.
- **Resultado esperado:** redirección a `/tasks` y la tarea aparece en la tabla.
- **Postcondiciones:** tarea persistida en H2 (sesión actual).

### TC04 — Editar tarea (CRUD)
- **Objetivo:** validar edición de una tarea existente.
- **Precondiciones:** usuario autenticado; existe al menos una tarea editable.
- **Pasos:** (1) Ir a `/tasks` (2) Seleccionar “Edit” (3) Cambiar valores (4) Guardar.
- **Resultado esperado:** tarea actualizada visible en listado (nuevo título/estado/prioridad).
- **Postcondiciones:** cambios persistidos en H2 (sesión actual).

### TC05 — Eliminar tarea (CRUD)
- **Objetivo:** validar eliminación de tarea.
- **Precondiciones:** usuario autenticado; existe una tarea a eliminar.
- **Pasos:** (1) Ir a `/tasks` (2) Click “Delete” en la tarea (3) Confirmar.
- **Resultado esperado:** la tarea desaparece de la tabla.
- **Postcondiciones:** tarea eliminada en H2 (sesión actual).

### TC06 — Filtrar por estado
- **Objetivo:** validar filtro por estado (COMPLETED).
- **Precondiciones:** usuario autenticado; existe al menos una tarea COMPLETED y una PENDING.
- **Pasos:** (1) Ir a `/tasks` (2) Seleccionar filtro “COMPLETED”.
- **Resultado esperado:** solo se muestran filas con estado COMPLETED.
- **Postcondiciones:** no se modifican datos; solo cambia la vista.

## 12. Matriz de trazabilidad (Requisitos ↔ Casos ↔ Herramienta)
| Requisito | Descripción | Caso(s) | Evidencia |
|---|---|---|---|
| RF-01 | Autenticación UI | TC01, TC02 | `reports/entrega-2/selenium/extent-report.html` |
| RF-02 | Dashboard | TC01 | `reports/entrega-2/selenium/extent-report.html` |
| RF-03 | Listar tareas | TC03–TC06 | `reports/entrega-2/selenium/extent-report.html` |
| RF-04 | Crear tarea | TC03 | `reports/entrega-2/selenium/extent-report.html` |
| RF-05 | Editar tarea | TC04 | `reports/entrega-2/selenium/extent-report.html` |
| RF-06 | Eliminar tarea | TC05 | `reports/entrega-2/selenium/extent-report.html` |
| RF-07 | Filtrar tareas | TC06 | `reports/entrega-2/selenium/extent-report.html` |
| RF-08 | API Tareas | (JUnit/MockMvc) | `reports/entrega-2/jacoco/index.html` |
| RNF-04 | Cobertura ≥ 80% | (JUnit) | `reports/entrega-2/jacoco/index.html` |
| RNF-05 | Rendimiento | (JMeter) | `reports/entrega-2/jmeter/html/index.html` |
