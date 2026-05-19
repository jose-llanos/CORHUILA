# AutoSpark - Plataforma de Automatización de Pruebas

## Descripción del Proyecto

AutoSpark es una plataforma web desarrollada para la gestión de servicios de lavado automotriz.  
El proyecto implementa una arquitectura Full Stack con backend en Spring Boot y frontend en Angular, integrando automatización de pruebas, análisis de calidad de código y despliegue Dockerizado mediante Jenkins.

El objetivo principal del proyecto fue construir un entorno completo de integración continua y pruebas automatizadas utilizando herramientas modernas de QA y DevOps.

> **Nota:** El proyecto cuenta con dos versiones relacionadas con la implementación de Jenkins, organizadas en diferentes ramas del repositorio.  
> La versión principal se encuentra en la rama actual, mientras que la segunda versión del proyecto está disponible en la rama `grupo2.1`.

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