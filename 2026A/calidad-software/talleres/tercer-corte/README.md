# 🏥 Clínica Veterinaria — Sistema de Gestión de Pruebas

> Proyecto Integrador — Calidad de Software 2026A  
> CORHUILA · Spring Boot · JUnit · Selenium · JMeter · SonarQube · Docker

---

## Tabla de contenidos

- [Descripción](#descripción)
- [Tecnologías](#tecnologías)
- [Requisitos previos](#requisitos-previos)
- [Ejecución con Docker](#ejecución-con-docker)
- [Ejecución local (sin Docker)](#ejecución-local-sin-docker)
- [Pruebas unitarias (JUnit)](#pruebas-unitarias-junit)
- [Pruebas Selenium](#pruebas-selenium)
- [Pruebas de rendimiento (JMeter)](#pruebas-de-rendimiento-jmeter)
- [Análisis SonarQube](#análisis-sonarqube)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Credenciales por defecto](#credenciales-por-defecto)

---

## Descripción

Sistema web de gestión para una clínica veterinaria que permite administrar **dueños**, **mascotas**, **citas** y **tratamientos**. El proyecto implementa un sistema integral de pruebas automatizadas siguiendo los estándares ISO/IEC 25010 e ISO/IEC 29119.

---

## Tecnologías

| Capa | Tecnología |
|------|-----------|
| Backend | Java 17, Spring Boot 3.2, Spring Security |
| Persistencia | Spring Data JPA, PostgreSQL 15, Hibernate |
| Frontend | Thymeleaf, HTML5, CSS3 |
| Pruebas unitarias | JUnit 5, Mockito, JaCoCo |
| Pruebas funcionales | Selenium WebDriver 4, Page Object Model |
| Pruebas de rendimiento | Apache JMeter 5.6 |
| Calidad de código | SonarQube 9.9 Community |
| Contenedores | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Documentación API | Springdoc OpenAPI (Swagger UI) |

---

## Requisitos previos

- **Docker** ≥ 24.0 y **Docker Compose** ≥ 2.0
- **Java JDK 17** (para ejecución local)
- **Maven 3.9+** (o usar `./mvnw`)
- **Google Chrome** (para pruebas Selenium)
- **Apache JMeter 5.6** (para pruebas de rendimiento)
- RAM mínima recomendada: **8 GB** (SonarQube requiere al menos 4 GB)

---

## Ejecución con Docker

### Paso 1 — Clonar el repositorio

```bash
git clone <URL-del-repositorio>
cd gestion-pruebas
```

### Paso 2 — Levantar todos los servicios

```bash
docker compose up -d
```

Esto levanta automáticamente:
- `veterinaria_db` — PostgreSQL en el puerto **5433**
- `veterinaria_app` — Aplicación Spring Boot en el puerto **8081**
- `sonar_db` — PostgreSQL para SonarQube
- `sonarqube` — Panel de calidad en el puerto **9000**

### Paso 3 — Verificar que los servicios estén activos

```bash
docker compose ps
```

### Paso 4 — Acceder a la aplicación

| Servicio | URL | Usuario | Contraseña |
|----------|-----|---------|------------|
| Aplicación web | http://localhost:8081 | admin | admin123 |
| SonarQube | http://localhost:9000 | admin | admin |
| Swagger UI | http://localhost:8081/swagger-ui.html | — | — |

### Detener los servicios

```bash
docker compose down
```

Para eliminar también los volúmenes (base de datos):

```bash
docker compose down -v
```

---

## Ejecución local (sin Docker)

### Paso 1 — Levantar solo la base de datos

```bash
docker compose up -d db
```

### Paso 2 — Ejecutar la aplicación

```bash
./mvnw spring-boot:run
```

O compilar primero:

```bash
./mvnw clean package -DskipTests
java -jar target/gestion-pruebas-0.0.1-SNAPSHOT.jar
```

---

## Pruebas unitarias (JUnit)

### Ejecutar todas las pruebas

```bash
./mvnw clean test
```

### Ejecutar con reporte de cobertura JaCoCo

```bash
./mvnw clean test jacoco:report
```

El reporte HTML se genera en:
```
target/site/jacoco/index.html
```

### Clases de prueba incluidas

| Clase | Pruebas | Cobertura objetivo |
|-------|---------|-------------------|
| `CitaServiceTest` | 11 casos | ≥ 80% |
| `DuenioServiceTest` | 7 casos | ≥ 80% |
| `MascotaServiceTest` | 9 casos | ≥ 80% |
| `TratamientoServiceTest` | 7 casos | ≥ 80% |

---

## Pruebas Selenium

> La aplicación debe estar corriendo en `http://localhost:8081` antes de ejecutar las pruebas Selenium.

### Requisitos

- Google Chrome instalado
- ChromeDriver compatible (WebDriverManager lo descarga automáticamente)

### Ejecutar pruebas Selenium (Chrome)

```bash
./mvnw test -Dtest=VeterinariaTest
```

### Ejecutar pruebas Selenium (Firefox)

```bash
./mvnw test -Dtest=VeterinariaTest -Dbrowser=firefox
```

### Casos de prueba incluidos

| # | Caso | Descripción |
|---|------|-------------|
| 1 | `caso1_crearNuevoDueno` | Crea un dueño con datos generados aleatoriamente |
| 2 | `caso2_crearNuevaMascota` | Crea una mascota asociada a un dueño |
| 3 | `caso3_validarCamposRequeridos` | Verifica validación HTML5 en formularios |
| 4 | `caso4_accesoModuloCitas` | Navega al módulo de citas y verifica la UI |
| 5 | `caso5_navegacionCompleta` | Navega por los 4 módulos del sistema |

---

## Pruebas de rendimiento (JMeter)

### Abrir el plan de pruebas en JMeter

```bash
jmeter -t src/test/jmeter/"Aggregate Report.jmx"
```

### Ejecutar en modo headless (sin GUI)

```bash
jmeter -n -t src/test/jmeter/"Aggregate Report.jmx" -l reports/jmeter/resultados.jtl -e -o reports/jmeter/reporte-html
```

### Escenarios configurados

| Escenario | Hilos (usuarios) | Ramp-up | Iteraciones |
|-----------|-----------------|---------|-------------|
| Normal | 5 | 10 s | 3 |
| Carga | 20 | 30 s | 5 |
| Estrés | 50 | 60 s | 5 |

---

## Análisis SonarQube

### Paso 1 — Levantar SonarQube

```bash
docker compose up -d sonarqube
```

Esperar ~2 minutos a que inicie. Verificar en http://localhost:9000

### Paso 2 — Crear proyecto en SonarQube

1. Entrar a http://localhost:9000 (admin / admin)
2. Crear proyecto manualmente con key: `clinica-veterinaria`
3. Generar un token de análisis y copiarlo

### Paso 3 — Ejecutar el scanner

```bash
./mvnw clean verify sonar:sonar \
  -Dsonar.projectKey=clinica-veterinaria \
  -Dsonar.host.url=http://localhost:9000 \
  -Dsonar.token=<TU_TOKEN>
```

---

## Estructura del repositorio

```
gestion-pruebas/
├── .github/
│   └── workflows/
│       └── ci.yml                  # Pipeline CI/CD GitHub Actions
├── src/
│   ├── main/
│   │   ├── java/com/corhuila/gestionpruebas/
│   │   │   ├── config/             # OpenAPI, Security
│   │   │   ├── controller/         # MVC Controllers
│   │   │   ├── model/              # Entidades JPA
│   │   │   ├── repository/         # Repositorios Spring Data
│   │   │   └── service/            # Lógica de negocio
│   │   └── resources/
│   │       ├── templates/          # Vistas Thymeleaf
│   │       └── application.properties
│   └── test/
│       └── java/com/corhuila/gestionpruebas/
│           ├── service/            # Pruebas unitarias JUnit
│           └── selenium/
│               ├── pages/          # Page Object Model
│               ├── tests/          # Tests Selenium
│               └── utils/          # Utilidades Selenium
│           └── jmeter/             # Plan JMeter (.jmx)
├── docs/
│   └── plan-de-pruebas.md          # Plan de pruebas ISO/IEC 29119
├── reports/                        # Reportes generados
├── docker-compose.yml
├── Dockerfile
├── sonar-project.properties
└── pom.xml
```

---

## Credenciales por defecto

| Servicio | Usuario | Contraseña |
|----------|---------|------------|
| Aplicación | admin | admin123 |
| PostgreSQL (app) | postgres | password123 |
| SonarQube | admin | admin |
| PostgreSQL (sonar) | sonar | sonar123 |

---

## CI/CD — GitHub Actions

El pipeline se ejecuta automáticamente en cada `push` a `main` o `develop`:

1. **Pruebas unitarias** — JUnit con cobertura JaCoCo (mínimo 80%)
2. **Build Docker** — Compila y verifica que la imagen arranca
3. **Análisis SonarQube** — Solo en rama `main` (requiere secrets configurados)

### Configurar secrets en GitHub

Para habilitar el análisis SonarQube en el pipeline:

```
Settings → Secrets and variables → Actions → New repository secret

SONAR_TOKEN     → token generado en SonarQube
SONAR_HOST_URL  → http://tu-servidor-sonarqube:9000
```
