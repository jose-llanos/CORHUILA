# MediCita - Sistema de Agendamiento de Citas Hospitalarias

Proyecto integrador de Calidad de Software 2026A.

Sistema web para la gestión y agendamiento de citas médicas, con módulos diferenciados para pacientes, médicos y administradores.

---

## Tecnologías utilizadas

| Capa | Tecnología |
|---|---|
| **Backend** | Java 26, Spring Boot 3.5.14, Spring Security, Spring Data JPA, PostgreSQL |
| **Frontend** | HTML5, Bootstrap 5, JavaScript vanilla |
| **Testing** | JUnit 5, Selenium WebDriver, JMeter, SonarQube |
| **Infraestructura** | Docker, Docker Compose, GitHub Actions |

---

## Estructura del repositorio

```
medicita-appweb/
├── backend/                    # API Spring Boot
│   ├── src/main/java/          # Código fuente Java
│   ├── src/main/resources/     # Configuración (application.yaml, data.sql)
│   └── pom.xml                 # Dependencias Maven
├── frontend/                   # SPA multi-page (Vite + nginx)
│   ├── index.html
│   ├── pages/                  # HTML por rol (auth, admin, doctor, patient)
│   ├── js/                     # Módulos ES (utils, lógica por rol)
│   ├── css/styles.css
│   ├── vite.config.js          # Dev server + proxy /api → backend:8080
│   ├── nginx.conf              # Sirve dist/ y proxea /api en producción
│   ├── Dockerfile              # Multi-stage: vite build → nginx
│   └── package.json
├── docker/                     # Archivos Docker del backend
│   ├── Dockerfile              # Build multi-stage del backend
│   ├── docker-compose.yml      # Stack completo (db + backend + frontend + sonarqube)
│   └── docker-compose.dev.yml  # Solo infraestructura para desarrollo local
├── tests/
│   ├── selenium/               # Pruebas de UI automatizadas
│   └── jmeter/                 # Planes de prueba de carga
├── docs/                       # Documentación del proyecto
├── reports/                    # Reportes de calidad y cobertura
├── .github/workflows/          # Pipelines de CI/CD
├── docker-compose.yml          # Acceso rápido al stack completo desde la raíz
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

### Opción A - Con Docker (recomendado)

```bash
git clone https://github.com/Kaleth2216/medicita-appweb.git
cd medicita-appweb
docker-compose up --build
```

Acceder al frontend en: http://localhost (nginx hace proxy de `/api` al backend en `:8080`)

### Opción B - Desarrollo local

```bash
# Levantar solo DB y SonarQube
docker-compose -f docker/docker-compose.dev.yml up -d

# Backend (terminal 1)
cd backend
./mvnw spring-boot:run

# Frontend (terminal 2) — primera vez: npm install
cd frontend
npm install
npm run dev
```

Frontend dev: http://localhost:5173 (Vite proxea `/api` a `http://localhost:8080`).

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

```bash
cd backend
./mvnw test
./mvnw jacoco:report
```

---

## Análisis SonarQube

```bash
cd backend
./mvnw sonar:sonar -Dsonar.host.url=http://localhost:9000
```

El dashboard de SonarQube estará disponible en: http://localhost:9000

---

## Integrantes del equipo

- <!-- Nombre 1 -->
- <!-- Nombre 2 -->
- <!-- Nombre 3 -->
- <!-- Nombre 4 -->
