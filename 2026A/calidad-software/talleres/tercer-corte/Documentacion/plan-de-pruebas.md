# Plan de Pruebas — Clínica Veterinaria
## Según ISO/IEC 29119-3: Documentación de Pruebas

**Proyecto:** Sistema de Gestión — Clínica Veterinaria  
**Versión:** 1.0  
**Asignatura:** Calidad de Software 2026A — CORHUILA  
**Fecha:** Mayo 2026  

---

## 1. Identificación del Plan

| Campo | Valor |
|-------|-------|
| Identificador | PP-CLINVET-2026-01 |
| Versión | 1.0 |
| Estado | Aprobado |
| Autor(es) | Equipo de desarrollo |
| Revisado por | Docente de Calidad de Software |

---

## 2. Introducción

### 2.1 Alcance del Plan de Pruebas

Este Plan de Pruebas cubre la validación integral del **Sistema de Gestión de Clínica Veterinaria**, una aplicación web desarrollada en Java con Spring Boot. El plan aplica los siguientes niveles de prueba:

- **Pruebas unitarias** sobre la capa de servicios (lógica de negocio)
- **Pruebas funcionales** de interfaz de usuario con Selenium WebDriver
- **Pruebas de rendimiento** con Apache JMeter
- **Análisis estático de calidad** con SonarQube

### 2.2 Elementos bajo prueba

| ID | Elemento | Versión |
|----|---------|---------|
| EL-01 | Módulo de Gestión de Dueños | 0.0.1 |
| EL-02 | Módulo de Gestión de Mascotas | 0.0.1 |
| EL-03 | Módulo de Gestión de Citas | 0.0.1 |
| EL-04 | Módulo de Gestión de Tratamientos | 0.0.1 |
| EL-05 | Sistema de autenticación | 0.0.1 |
| EL-06 | API REST / Endpoints MVC | 0.0.1 |

### 2.3 Elementos fuera del alcance

- Pruebas de seguridad avanzada (penetration testing)
- Pruebas de compatibilidad con navegadores distintos a Chrome y Firefox
- Pruebas de accesibilidad (WCAG)
- Pruebas en dispositivos móviles

---

## 3. Contexto de las Pruebas

### 3.1 Ambiente de prueba

**Ambiente de desarrollo:**

| Componente | Configuración |
|-----------|---------------|
| SO | Ubuntu 24 / Windows 11 |
| JDK | OpenJDK 17 |
| Framework | Spring Boot 3.2.5 |
| Base de datos | PostgreSQL 15 |
| Contenedores | Docker 24, Docker Compose 2 |

**Herramientas de prueba:**

| Herramienta | Versión | Propósito |
|-------------|---------|-----------|
| JUnit | 5 (Jupiter) | Pruebas unitarias |
| Mockito | 5.x | Mocking de dependencias |
| JaCoCo | 0.8.11 | Cobertura de código |
| Selenium WebDriver | 4.23.0 | Pruebas funcionales UI |
| WebDriverManager | 5.9.2 | Gestión de drivers |
| Apache JMeter | 5.6.3 | Pruebas de rendimiento |
| SonarQube | 9.9 Community | Análisis estático |

### 3.2 Criterios de entrada

- Código fuente compilado sin errores
- Base de datos PostgreSQL disponible y con esquema creado
- Aplicación desplegada y accesible en `http://localhost:8081`
- Navegadores Chrome y Firefox instalados (para Selenium)

### 3.3 Criterios de salida

- Cobertura de código ≥ 80% en clases de servicio
- 0 pruebas unitarias fallidas
- 5/5 casos Selenium pasados en Chrome
- 5/5 casos Selenium pasados en Firefox
- Análisis SonarQube completado con rating de seguridad ≥ B
- Reporte JMeter generado para los 3 escenarios

### 3.4 Criterios de suspensión y reanudación

**Suspensión:** Se suspenden las pruebas si:
- La aplicación no puede arrancar
- La base de datos no responde
- Más del 30% de los casos unitarios fallan en la compilación

**Reanudación:** Se reanudan una vez corregidos los defectos bloqueantes.

---

## 4. Estrategia de Pruebas

### 4.1 Enfoque de prueba

Se aplica un enfoque de **pruebas en capas**, siguiendo la pirámide de testing:

```
        /\
       /UI\        ← Selenium (5 casos mínimo)
      /----\
     /  Int \      ← Integración Spring (opcional)
    /--------\
   / Unitarias\   ← JUnit + Mockito (80% cobertura)
  /____________\
```

### 4.2 Niveles de prueba

#### Nivel 1: Pruebas Unitarias

- **Objetivo:** Verificar la lógica de negocio de cada servicio de forma aislada
- **Técnica:** Caja blanca — se conoce la implementación interna
- **Herramienta:** JUnit 5 + Mockito
- **Cobertura mínima:** 80% en líneas de código de los servicios
- **Clases bajo prueba:** `CitaService`, `DuenioService`, `MascotaService`, `TratamientoService`

#### Nivel 2: Pruebas Funcionales (UI)

- **Objetivo:** Verificar flujos de usuario completos desde el navegador
- **Técnica:** Caja negra — se prueba el comportamiento externo
- **Herramienta:** Selenium WebDriver 4 + Page Object Model
- **Navegadores:** Chrome (principal), Firefox (secundario)
- **Mínimo:** 5 casos de prueba cubriendo flujos críticos

#### Nivel 3: Pruebas de Rendimiento

- **Objetivo:** Evaluar el comportamiento del sistema bajo carga
- **Herramienta:** Apache JMeter 5.6
- **Escenarios:** Normal (5 usuarios), Carga (20 usuarios), Estrés (50 usuarios)
- **Métricas objetivo:**
  - Tiempo de respuesta promedio < 2 segundos (escenario normal)
  - Tasa de error < 1% (escenario normal y carga)
  - Throughput > 10 req/s (escenario normal)

#### Nivel 4: Análisis Estático de Calidad

- **Objetivo:** Detectar bugs, vulnerabilidades y deuda técnica
- **Herramienta:** SonarQube 9.9 Community
- **Métricas objetivo:**
  - Bugs: 0 críticos, ≤ 5 mayores
  - Vulnerabilidades de seguridad: 0
  - Code smells: densidad < 5%
  - Duplicación de código: < 10%
  - Rating de mantenibilidad: A o B

### 4.3 Técnicas de diseño de casos de prueba

| Técnica | Aplicación |
|---------|-----------|
| Partición de equivalencia | Valores válidos vs inválidos en campos |
| Valores límite | Campos nulos, vacíos, longitud mínima/máxima |
| Flujo de control | Caminos en los métodos de servicio |
| Casos de uso | Escenarios completos de usuario en Selenium |

---

## 5. Casos de Prueba

### 5.1 Pruebas unitarias — CitaService

| ID | Nombre | Precondición | Pasos | Resultado esperado |
|----|--------|-------------|-------|-------------------|
| CT-01 | guardar cita válida | Cita con motivo, fecha y mascota | Llamar `guardar(cita)` | Retorna cita guardada |
| CT-02 | guardar sin motivo | Cita sin motivo | Llamar `guardar(cita)` | Lanza `IllegalArgumentException` |
| CT-03 | guardar motivo vacío | Motivo = "" | Llamar `guardar(cita)` | Lanza `IllegalArgumentException` |
| CT-04 | guardar sin fecha | Cita sin fecha | Llamar `guardar(cita)` | Lanza `IllegalArgumentException` |
| CT-05 | asignar estado por defecto | Estado null | Llamar `guardar(cita)` | Estado = "PENDIENTE" |
| CT-06 | obtener todas las citas | BD con 1 cita | Llamar `obtenerTodas()` | Lista de 1 elemento |
| CT-07 | buscar por ID existente | Cita con ID=1 | Llamar `buscarPorId(1)` | Retorna la cita |
| CT-08 | buscar por ID inexistente | No existe ID=99 | Llamar `buscarPorId(99)` | Retorna null |
| CT-09 | cambiar estado válido | Cita con ID=1 | Cambiar a "FINALIZADA" | Estado actualizado |
| CT-10 | cambiar estado ID inválido | No existe ID=99 | Cambiar estado | Lanza excepción |
| CT-11 | eliminar cita | Cita con ID=1 | Llamar `eliminar(1)` | `deleteById` ejecutado |

### 5.2 Pruebas funcionales — Selenium

| ID | Caso | Módulo | Flujo | Resultado esperado |
|----|------|--------|-------|-------------------|
| SE-01 | Crear dueño | Dueños | Ir a nuevo → Llenar form → Guardar | Dueño aparece en tabla |
| SE-02 | Crear mascota | Mascotas | Crear dueño → Crear mascota | Mascota aparece en lista |
| SE-03 | Validar campos requeridos | Formularios | Submit sin datos | Validación HTML5 activa |
| SE-04 | Acceso módulo citas | Citas | Navegar a /citas | Página carga correctamente |
| SE-05 | Navegación completa | Todos | Visitar 4 módulos | Todos responden con 200 |

### 5.3 Pruebas de rendimiento — JMeter

| ID | Escenario | Usuarios | Ramp-up | Iteraciones | Métricas a medir |
|----|-----------|----------|---------|-------------|-----------------|
| JM-01 | Normal | 5 | 10 s | 3 | Tiempo respuesta, throughput |
| JM-02 | Carga | 20 | 30 s | 5 | Tiempo respuesta, % error |
| JM-03 | Estrés | 50 | 60 s | 5 | Punto de saturación, errores |

**Endpoints evaluados en cada escenario:**
- `GET /` — Página principal
- `GET /duenios` — Listado de dueños
- `GET /mascotas` — Listado de mascotas
- `GET /citas` — Listado de citas

---

## 6. Gestión de Defectos

### 6.1 Clasificación de severidad

| Severidad | Descripción | Tiempo máximo de corrección |
|-----------|-------------|---------------------------|
| Crítica | El sistema no arranca o pierde datos | Inmediato |
| Alta | Funcionalidad principal no opera | 24 horas |
| Media | Funcionalidad secundaria afectada | 48 horas |
| Baja | Defecto cosmético o menor | Próxima versión |

### 6.2 Ciclo de vida del defecto

```
Nuevo → Asignado → En corrección → Verificación → Cerrado
                                 ↓
                              Rechazado → Reabierto
```

---

## 7. Métricas de Calidad Objetivo

| Métrica | Fórmula | Objetivo |
|---------|---------|---------|
| Cobertura de código | Líneas ejecutadas / Total líneas × 100 | ≥ 80% |
| Densidad de defectos | Defectos / KLOC | < 5 defectos/KLOC |
| Tasa de éxito de pruebas | Pruebas pasadas / Total × 100 | ≥ 95% |
| Tiempo respuesta promedio | Promedio de todas las solicitudes | < 2 s (normal) |
| Tasa de error JMeter | Solicitudes fallidas / Total × 100 | < 1% (normal/carga) |
| Rating seguridad SonarQube | Clasificación A-E | ≥ B |

---

## 8. Roles y Responsabilidades

| Rol | Responsabilidades |
|-----|------------------|
| Desarrollador/Tester | Diseñar y ejecutar pruebas unitarias y Selenium |
| QA Lead | Revisar plan de pruebas, análisis SonarQube |
| Equipo completo | Pruebas JMeter, documentación de resultados |

---

## 9. Cronograma de Pruebas

| Actividad | Duración estimada |
|-----------|-----------------|
| Diseño de pruebas unitarias | 1 día |
| Ejecución pruebas unitarias + JaCoCo | 30 minutos |
| Diseño casos Selenium | 1 día |
| Ejecución Selenium (Chrome + Firefox) | 1 hora |
| Configuración y ejecución JMeter | 2 horas |
| Análisis SonarQube | 1 hora |
| Documentación de resultados | 1 día |

---

## 10. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|-----------|
| Selenium falla por cambios en UI | Media | Alta | Usar selectores robustos (name, id) |
| SonarQube no arranca por falta de RAM | Alta | Media | Aumentar heap o usar SonarCloud gratuito |
| Cobertura < 80% | Baja | Alta | Agregar casos de prueba adicionales |
| JMeter no puede autenticarse | Media | Alta | Configurar manejador de cookies y CSRF |

---

## 11. Entregables de Prueba

| Entregable | Ubicación | Responsable |
|-----------|----------|-------------|
| Código de pruebas JUnit | `src/test/java/.../service/` | Equipo |
| Reporte JaCoCo | `target/site/jacoco/` | Maven |
| Código Selenium + POM | `src/test/java/.../selenium/` | Equipo |
| Plan JMeter | `src/test/jmeter/*.jmx` | Equipo |
| Reporte JMeter HTML | `reports/jmeter/` | JMeter |
| Reporte SonarQube | `reports/sonarqube/` / Dashboard | SonarQube |
| Este plan de pruebas | `docs/plan-de-pruebas.md` | Equipo |

---

## 12. Aprobación

| Nombre | Rol | Firma | Fecha |
|--------|-----|-------|-------|
| | Autor principal | | Mayo 2026 |
| | Revisor | | Mayo 2026 |
| | Docente | | Mayo 2026 |
