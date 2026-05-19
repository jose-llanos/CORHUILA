# AutoSpark - Plataforma de Automatización de Pruebas
> *Nota:* El proyecto cuenta con dos versiones relacionadas con la implementación de Jenkins, organizadas en diferentes ramas del repositorio.  
> La versión principal se encuentra en la rama `grupo2`, mientras que la segunda versión del proyecto está disponible en esta rama (`grupo2.1`).


## Descripción del Proyecto

AutoSpark es una plataforma web desarrollada para la gestión de servicios de lavado automotriz.  
El proyecto implementa una arquitectura Full Stack con backend en Spring Boot y frontend en Angular, integrando automatización de pruebas, análisis de calidad de código y despliegue Dockerizado mediante Jenkins.

El objetivo principal del proyecto fue construir un entorno completo de integración continua y pruebas automatizadas utilizando herramientas modernas de QA y DevOps.

---

# Tecnologías Utilizadas

## Backend
- Java 21
- Spring Boot
- Maven
- JUnit 5
- JaCoCo

## Frontend
- Angular

## Base de Datos
- MySQL 

## Automatización y QA
- Selenium WebDriver
- JMeter
- SonarQube

## DevOps
- Docker
- Docker Compose
- Jenkins
- GitHub Actions

---

# Arquitectura del Proyecto

```text
automatizacion-pruebas/
├── app/                        # Backend Spring Boot
│   └── pom.xml
├── frontend/                   # Frontend Angular
├── docker/                     # Docker Compose y Dockerfiles
├── docs/                       # Documentación y evidencias
├── reports/                    # Reportes generados
├── tests/
│   ├── selenium/               # Pruebas funcionales
│   └── jmeter/                 # Pruebas de rendimiento
├── .github/workflows/          # GitHub Actions
├── Jenkinsfile                 # Pipeline Jenkins
├── .gitignore
└── README.md
```

---

# Funcionalidades Implementadas

- Gestión de usuarios
- Inicio de sesión
- Registro de usuarios
- Recuperación de contraseña
- Gestión de servicios
- Creación de reservas
- Consulta de reservas
- Validaciones de backend
- Automatización de pruebas funcionales
- Automatización de pruebas de rendimiento


---

# Contenedores Docker

El sistema utiliza los siguientes contenedores:

| Contenedor | Función |
|---|---|
| autospark_backend | API Spring Boot |
| autospark_frontend | Frontend Angular + Nginx |
| autospark_mysql | Base de datos MySQL |
| autospark_sonarqube | Análisis de calidad |
| autospark_jenkins | Pipeline CI/CD |

---

# Ejecución del Proyecto

## 1. Clonar repositorio

```bash
git clone <URL_REPOSITORIO>
cd automatizacion-pruebas
```

---

## 2. Levantar contenedores

```bash
cd docker
docker compose up -d --build
```

---

## 3. Acceder a servicios

| Servicio | URL |
|---|---|
| Frontend | http://localhost:4200 |
| Backend | http://localhost:8080 |
| Jenkins | http://localhost:8081 |
| SonarQube | http://localhost:9000 |

---

# Pipeline Jenkins

El pipeline automatiza:

- Compilación backend
- Ejecución de pruebas unitarias
- Generación de cobertura JaCoCo
- Análisis SonarQube
- Despliegue Docker
- Inserción de datos de prueba
- Ejecución de pruebas JMeter
- Ejecución de pruebas Selenium
- Validación de reportes

---

# Reportes Generados

## JaCoCo
```text
app/target/site/jacoco/index.html
```

## JMeter
```text
reports/jmeter/html/index.html
```

## Selenium
```text
tests/selenium/test-output/
```

## SonarQube
```text
http://localhost:9000
```


# Integración Continua

## GitHub Actions
Se utiliza para validar automáticamente el backend mediante pruebas unitarias en cada push o pull request.

## Jenkins
Se utiliza como pipeline principal de integración continua y automatización de pruebas.

---

