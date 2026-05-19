# MediCita v2 — Sistema de Agendamiento de Citas Hospitalarias

> **Versión 2** — Esta versión aplica el sistema integral de pruebas QA/QC exigido por el Proyecto Integrador de Calidad de Software 2026A: pruebas unitarias (JUnit + JaCoCo), funcionales (Selenium + POM), de rendimiento (JMeter), análisis estático (SonarQube), Docker Compose y CI/CD con GitHub Actions. Para la versión base sin testing aplicado, ver `../proyectoIntegrador-v1/`.

Sistema web para la gestión y agendamiento de citas médicas, con módulos diferenciados para pacientes, médicos y administradores.

---

## Integrantes del equipo

| Nombre | Grupo |
|---|---|
| Anderson Kaleth Camacho Cabezas | 11 |
| Edward Alexis Serrudo Ariza | 11 |

---

## Tecnologías utilizadas

| Capa | Tecnología |
|---|---|
| **Backend** | Java 21, Spring Boot 3.5.14, Spring Security, Spring Data JPA, PostgreSQL 15 |
| **Frontend** | HTML5, Bootstrap 5, JavaScript vanilla, nginx |
| **Pruebas unitarias** | JUnit 5, Mockito, AssertJ, JaCoCo 0.8.12 |
| **Pruebas funcionales** | Selenium WebDriver 4.27, Page Object Model, Chrome headless |
| **Pruebas de rendimiento** | Apache JMeter 5.6.2 (vía jmeter-maven-plugin 3.8.0) |
| **Calidad de código** | SonarQube 10 Community Edition |
| **Infraestructura** | Docker, Docker Compose, GitHub Actions |

---

## Estructura del repositorio

```
proyectoIntegrador-v2/
├── backend/                    # API Spring Boot
│   ├── src/main/java/          # Código fuente Java
│   ├── src/main/resources/     # Configuración (application.yaml, data.sql)
│   └── pom.xml                 # Dependencias Maven + JaCoCo + SonarQube
├── frontend/                   # SPA multi-page (Vite + nginx)
│   ├── pages/                  # HTML por rol (auth, admin, doctor, patient)
│   ├── js/                     # Módulos ES (utils, lógica por rol)
│   ├── css/styles.css
│   ├── nginx.conf
│   └── Dockerfile
├── docker/
│   ├── Dockerfile              # Build multi-stage del backend
│   ├── docker-compose.yml      # Stack completo (db + backend + frontend + sonarqube)
│   └── docker-compose.dev.yml  # Solo infraestructura para desarrollo local
├── tests/
│   ├── selenium/               # Pruebas funcionales con Page Object Model (TC01–TC07)
│   └── jmeter/                 # Plan de carga (Normal 5u / Alta 30u / Estrés 100u)
├── reports/
│   ├── jacoco/                 # Reporte HTML de cobertura (index.html)
│   ├── jmeter/                 # Resultados CSV de ejecución
│   └── sonarqube/              # Ver capturas en docs/
├── docs/                       # Reporte QA/QC (reporte-qa.html)
├── .github/workflows/
│   └── ci-medicita.yml         # Pipeline CI/CD (4 jobs en paralelo)
└── README.md
```

---

## Requisitos previos

- Java 21+
- Maven 3.8+
- Docker y Docker Compose
- Git

---

## Instalación y ejecución

### Opción A — Con Docker (recomendado)

```bash
# Desde la carpeta proyectoIntegrador-v2/
docker compose -f docker/docker-compose.yml up --build
```

Acceder en: `http://localhost` (nginx hace proxy de `/api` al backend en `:8080`)

### Opción B — Desarrollo local

```bash
# Levantar solo DB y SonarQube
docker compose -f docker/docker-compose.dev.yml up -d

# Backend (terminal 1)
cd backend
./mvnw spring-boot:run

# Frontend (terminal 2) — primera vez: npm install
cd frontend
npm install
npm run dev
```

Frontend dev: `http://localhost:5173`

---

## Credenciales por defecto

| Rol | Email | Contraseña |
|---|---|---|
| Admin | admin@medicita.com | Admin2026* |

---

## Módulos del sistema

- **Módulo Paciente**: registro, agendamiento de citas, historial de consultas.
- **Módulo Médico**: gestión de horario semanal, atención de citas, solicitud de permisos.
- **Módulo Admin**: gestión de médicos y especialidades, aprobación de permisos.

---

## Ejecutar pruebas

### Pruebas unitarias + cobertura JaCoCo

```bash
cd backend
./mvnw verify
# Reporte en: backend/target/site/jacoco/index.html
```

### Pruebas funcionales Selenium (requiere stack Docker activo)

```bash
# 1. Levantar el stack
docker compose -f docker/docker-compose.yml up -d --build

# 2. Ejecutar los 7 casos TC01–TC07
cd tests/selenium
mvn test -Dapp.url=http://localhost
```

### Pruebas de rendimiento JMeter

```bash
# Requiere backend corriendo en localhost:8080
cd tests/jmeter
mvn verify
# Resultados en: tests/jmeter/results/
```

### Análisis SonarQube

```bash
# 1. Levantar SonarQube (incluido en docker-compose.yml, puerto 9000)
docker compose -f docker/docker-compose.yml up sonarqube -d

# 2. Ejecutar análisis
cd backend
./mvnw verify sonar:sonar \
  -Dsonar.host.url=http://localhost:9000 \
  -Dsonar.token=<TOKEN>
# Dashboard: http://localhost:9000 → proyecto "medicita"
```

---

## CI/CD — GitHub Actions

El pipeline se activa automáticamente en cada push a la rama `kaleth` con cambios dentro de este proyecto. Ejecuta 4 jobs en paralelo:

| Job | Herramienta | Resultado |
|---|---|---|
| Unit Tests & JaCoCo | JUnit 5 + JaCoCo | 96 tests · 91% cobertura en servicios |
| Selenium Functional Tests | Selenium WebDriver | 7/7 casos TC01–TC07 |
| JMeter Performance Tests | Apache JMeter | 3 escenarios · 0% errores en endpoints de negocio |
| SonarQube Code Analysis | SonarQube 10 | Quality Gate: OK · Rating A |

Los artefactos (reporte JaCoCo, resultados JMeter, reporte SonarQube) quedan disponibles en la pestaña **Actions → Artifacts** de GitHub por 7 días.
