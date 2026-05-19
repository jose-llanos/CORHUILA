# Reporte de Ejecución — Task Manager
**Materia:** Calidad de Software 2026A  
**Proyecto:** Task Manager  
**Versión:** 1.0.0  
**Fecha:** 2026-05-19

## 1. Resumen
Se ejecutó el ciclo de aseguramiento de calidad con:
- Pruebas unitarias/integración (JUnit 5) con build exitoso.
- Cobertura con JaCoCo (reporte HTML/XML).
- Análisis estático en SonarQube (Quality Gate en estado Passed).
- Pruebas de rendimiento con JMeter (reporte HTML).
- Pruebas funcionales UI con Selenium (reporte ExtentReports).

## 2. Evidencias generadas (rutas)
### 2.1 JaCoCo
- Reporte HTML: `reports/entrega-2/jacoco/index.html`
- Reporte XML: `reports/entrega-2/jacoco/jacoco.xml`
- CSV: `reports/entrega-2/jacoco/jacoco.csv`

Cobertura agregada (según `jacoco.csv`):
- Instruction coverage: **92.27%**
- Line coverage: **96.80%**
- Branch coverage: **86.11%**

### 2.2 SonarQube
- Dashboard local: `http://localhost:9000/dashboard?id=taskmanager`

### 2.3 JMeter
- Resultados: `reports/entrega-2/jmeter/results.jtl`
- Reporte HTML: `reports/entrega-2/jmeter/html/index.html`

### 2.4 Selenium (ExtentReports)
- Reporte HTML: `reports/entrega-2/selenium/extent-report.html`

## 3. Resultados por tipo de prueba
### 3.1 Unitarias/Integración (JUnit)
- Estado: **OK**
- Artefactos: surefire reports en `app/target/surefire-reports/`

### 3.2 UI (Selenium)
Casos ejecutados:
- TC01–TC06

Estado:
- **OK** (reporte disponible en `reports/entrega-2/selenium/extent-report.html`)

### 3.3 Rendimiento (JMeter)
- Estado: **OK**
- Reporte HTML generado correctamente.

## 4. Incidencias y correcciones aplicadas (resumen)
- Errores 500 en UI por plantillas Thymeleaf: corregidos en vistas.
- Autorización de análisis SonarQube: se ejecuta con token (variable de entorno) para evitar fallos de permisos.
- JMeter: ajuste de host para ejecución dentro de red Docker (`BASE_HOST=app`).
- Selenium: descarga de imágenes dependiente de red; se usa `selenium/standalone-chrome` y esperas explícitas para estabilidad.

## 5. Defectos registrados (defect log)
| ID | Tipo | Descripción | Severidad | Estado |
|---|---|---|---|---|
| DEF-01 | UI | Error 500 en `/tasks` por plantilla Thymeleaf (valor vacío en option) | Alta | Corregido |
| DEF-02 | UI | Error 500 en `/tasks/new` por expresión en formulario | Alta | Corregido |
| DEF-03 | Automatización | NPE en reporte ExtentReports al iniciar cada test Selenium | Media | Corregido |
| DEF-04 | Automatización | Flakiness Selenium por navegación/esperas (timeout esperando tabla) | Media | Corregido |
| DEF-05 | Integración | Fallo de prueba por redirección absoluta vs relativa en login | Baja | Corregido |

Evidencia de ejecución posterior a correcciones:
- JaCoCo: `reports/entrega-2/jacoco/index.html`
- Selenium: `reports/entrega-2/selenium/extent-report.html`

## 6. Trazabilidad (resumen)
La matriz de trazabilidad Requisitos ↔ Casos ↔ Evidencias se encuentra en:
- `docs/Plan_de_Pruebas.md` (sección “Matriz de trazabilidad”)

## 7. Métricas
### 7.1 Cobertura (JaCoCo)
- Instruction coverage: **92.27%**
- Line coverage: **96.80%**
- Branch coverage: **86.11%**

### 7.2 Densidad de defectos
Definición: defectos detectados / KLOC del código productivo.
- LOC (app/src/main/java): **899**
- KLOC: **0.899**
- Defectos registrados (DEF-01 a DEF-05): **5**
- Densidad de defectos: **5.56 defectos/KLOC**

### 7.3 MTTR (Mean Time To Repair)
Para MTTR se requiere registrar (por defecto) fecha/hora de detección y fecha/hora de corrección por defecto.
- MTTR: **N/D** (no se registraron tiempos de inicio/fin por defecto en un sistema de tickets)

## 8. Conclusión
El proyecto cumple el flujo de QA planteado: genera evidencias de pruebas (JUnit/JaCoCo, JMeter, Selenium) y análisis estático (SonarQube) en un entorno reproducible con Docker.
