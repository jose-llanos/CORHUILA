# MAP Parking — Proyecto Integrador (Calidad de Software)

Sistema web de gestión de estacionamiento con suite de pruebas automatizadas (JUnit, Selenium, JMeter), análisis SonarQube y orquestación Docker.

## Estructura del repositorio

```
app/
├── Backend/Backend/     # API Spring Boot (Java 17)
├── Front/frontend/      # SPA Angular
├── Docker/              # docker-compose (MySQL, app, SonarQube)
├── tests/
│   ├── jmeter/          # Pruebas de rendimiento (.jmx)
│   └── selenium/        # Pruebas UI (POM + JUnit 5)
├── reports/             # Salidas JMeter, Selenium, cobertura
├── docs/                # Documentación ISO/IEC 29119 (entrega 3)
└── .github/workflows/   # CI/CD (GitHub Actions)
```

## Requisitos

- JDK 17+, Maven 3.9+
- Node.js 20+ y npm (frontend)
- Docker Desktop
- Apache JMeter 5.x (rendimiento)
- Brave Browser (Selenium local)

## Puesta en marcha

### 0. Configuración del backend (primera vez)

```bash
cd Backend/Backend/src/main/resources
copy application.properties.example application.properties
# Edita application.properties con tus credenciales de correo si usas recuperación de contraseña
```

### 1. Backend + base de datos + SonarQube

```bash
cd Docker
docker compose up -d
```

- API: http://localhost:8080  
- SonarQube: http://localhost:9000  
- Mysql: http://localhost:3306

### 2. Frontend

```bash
cd Front/frontend
npm install
npm start o ng serve
```

- UI: http://localhost:4200  

## Ejecutar pruebas

| Tipo | Comando | Documentación |
|------|---------|---------------|
| Unitarias + JaCoCo | `cd Backend/Backend && mvn verify` | Cobertura ≥ 80 % en lógica de negocio |
| Selenium (Brave) | `cd tests/selenium && ./run-selenium.ps1` | [tests/selenium/README.md](tests/selenium/README.md) |
| JMeter | `cd tests/jmeter && ./run-jmeter.ps1 o mvn test -Dbrowser=brave` | [tests/jmeter/README.md](tests/jmeter/README.md) |

## CI/CD

El workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) ejecuta en cada push/PR:

1. **JUnit** con verificación JaCoCo  
2. **Selenium** (Chrome headless en GitHub Actions; localmente usa **Brave**)

## SonarQube (local)

Con el stack Docker levantado, desde `Backend/Backend`:

```bash
mvn verify sonar:sonar -Dsonar.host.url=http://localhost:9000 -Dsonar.token=TU_TOKEN
```

## Usuario de prueba Selenium

Creado automáticamente por API si no existe:

- Email: `selenium.admin@test.com`  
- Contraseña: `Test1234!`  
- Rol: `Administrador`  

## Entregables del curso

| Entrega | Contenido |
|---------|-----------|
| 1 | App, Docker, README |
| 2 | JUnit, Selenium, JMeter, SonarQube, GitHub Actions |
| 3 | Plan de pruebas, reportes, informe PDF (APA 7) |
| 4 | Sustentación (10 diapositivas, 15 min) |

## Equipo / asignatura

Calidad de Software — 2026A
