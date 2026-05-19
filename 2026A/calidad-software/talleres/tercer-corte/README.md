# TaskApp — Sistema de Gestión de Tareas
 
> Proyecto Integrador — Calidad de Software 2026A · CORHUILA · Grupo 11
 
Aplicación web para la gestión colaborativa de proyectos y tareas, desarrollada en Spring Boot con Java 21. Este repositorio contiene tanto el código de producción como el sistema integral de aseguramiento y control de calidad (QA/QC): pruebas unitarias, pruebas funcionales de interfaz, pruebas de rendimiento, análisis estático de código y orquestación con CI/CD.
 
---
 
## Tabla de Contenidos
 
- [Descripción del Proyecto](#descripción-del-proyecto)
- [Stack Tecnológico](#stack-tecnológico)
- [Arquitectura](#arquitectura)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Requisitos Previos](#requisitos-previos)
- [Instalación y Ejecución](#instalación-y-ejecución)
- [Funcionalidades](#funcionalidades)
- [Sistema de Pruebas](#sistema-de-pruebas)
  - [Pruebas Unitarias (JUnit)](#pruebas-unitarias-junit)
  - [Pruebas Funcionales (Selenium)](#pruebas-funcionales-selenium)
  - [Pruebas de Rendimiento (JMeter)](#pruebas-de-rendimiento-jmeter)
  - [Análisis de Calidad (SonarQube)](#análisis-de-calidad-sonarqube)
- [Integración Continua (Jenkins)](#integración-continua-jenkins)
- [Endpoints de la API](#endpoints-de-la-api)
- [Decisiones de Diseño](#decisiones-de-diseño)
- [Autores](#autores)
---
 
## Descripción del Proyecto
 
TaskApp permite a los usuarios crear proyectos colaborativos, invitar a otros miembros y gestionar tareas en un tablero estilo Kanban (Pendiente · En Progreso · Terminado). Cada proyecto tiene un *owner* (creador) y puede tener múltiples miembros, con permisos diferenciados:
 
- **Owner:** control total sobre el proyecto, miembros y tareas.
- **Miembro:** puede ver el proyecto, crear y editar tareas, y cambiar el estado de las tareas asignadas a él.
El sistema se entrega junto con un plan integral de pruebas alineado a ISO/IEC 29119 y al modelo de calidad ISO/IEC 25010.
 
---
 
## Stack Tecnológico
 
### Backend
| Componente | Versión |
|---|---|
| Java | 21 |
| Spring Boot | 4.0.6 |
| Spring Data JPA | (incluida) |
| Spring Security | (incluida) |
| Spring Validation | (incluida) |
| JJWT | 0.12.6 |
| Lombok | (última) |
| PostgreSQL | 15 |
| Maven Wrapper | `./mvnw` |
 
### Frontend
HTML5 + CSS3 + JavaScript vanilla (sin framework). Las vistas son servidas estáticamente y consumen la API REST.
 
### Pruebas y Calidad
| Herramienta | Propósito |
|---|---|
| JUnit 5 + Mockito | Pruebas unitarias |
| JaCoCo | Cobertura de código |
| Selenium WebDriver 4.x | Pruebas funcionales de UI |
| Selenium Grid (Docker) | Ejecución cross-browser |
| JMeter 5.x | Pruebas de carga y estrés |
| SonarQube CE | Análisis estático de código |
| Jenkins | CI/CD |
| Docker + Docker Compose | Orquestación |
 
---
 
## Arquitectura
 
```
┌─────────────────────────────────────────────────────────┐
│                    Cliente (Navegador)                  │
│              HTML + CSS + JS Vanilla                    │
└────────────────────────┬────────────────────────────────┘
                         │ HTTPS/REST + JWT
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Spring Boot 4.0.6 (Java 21)                │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────────┐  │
│  │  Controller  │→ │   Service   │→ │   Repository   │  │
│  └──────────────┘  └─────────────┘  └────────────────┘  │
│         ▲                                  │            │
│         │  JwtAuthenticationFilter         │            │
│         │  + SecurityConfig                ▼            │
└─────────┴────────────────────────┬───────────────────────┘
                                   │ JDBC
                                   ▼
                          ┌──────────────────┐
                          │  PostgreSQL 15   │
                          └──────────────────┘
```
 
El principal del `SecurityContext` es la entidad `User` directamente. El `JwtAuthenticationFilter` resuelve el usuario desde `UserRepository` sin pasar por un `UserDetailsService`.
 
---
 
## Estructura del Repositorio
 
```
tercer-corte/
├── app/                          # Aplicación Spring Boot
│   ├── src/main/java/com/tasks/app/
│   │   ├── controller/           # AuthController, ProjectController, TaskController
│   │   ├── service/              # UserService, ProjectService, TaskService
│   │   ├── repository/           # UserRepository, ProjectRepository, TaskRepository, ProjectMemberRepository
│   │   ├── entity/               # User, Project, Task, ProjectMember, TaskStatus
│   │   ├── dto/
│   │   │   ├── request/          # DTOs de entrada
│   │   │   └── response/         # DTOs de salida
│   │   ├── exception/            # ConflictException, ForbiddenException, etc.
│   │   └── security/             # JwtService, JwtAuthenticationFilter, SecurityConfig
│   ├── src/main/resources/
│   │   ├── static/               # index.html, dashboard.html, css/, js/
│   │   └── application.yml
│   ├── src/test/java/com/tasks/app/
│   │   ├── service/              # Tests unitarios (UserServiceTest, ProjectServiceTest, TaskServiceTest)
│   │   ├── security/             # JwtServiceTest
│   │   ├── repository/           # RepositoryIntegrityTest (TU04)
│   │   └── suite/                # Suites JUnit (Authentication, ProjectManagement, etc.)
│   ├── pom.xml
│   └── mvnw
├── tests/
│   ├── selenium/                 # Pruebas funcionales (Page Object Model)
│   └── jmeter/                   # Planes de prueba (.jmx)
├── docker/
│   ├── docker-compose.yml        # Orquestación: app, db, sonarqube, jenkins
│   ├── Dockerfile.app
│   └── Dockerfile.jenkins
├── reports/                      # Reportes generados (JaCoCo, JMeter, SonarQube)
├── docs/                         # Plan de pruebas, especificaciones, evidencia
├── .github/workflows/            # GitHub Actions (CI)
└── README.md
```
 
---
 
## Requisitos Previos
 
- **Docker** ≥ 24.x
- **Docker Compose** ≥ 2.x
- **Git**
- **RAM:** mínimo 8 GB (Docker + SonarQube son exigentes)
- **Disco:** mínimo 10 GB libres
> ⚠️ Java no es necesario en la máquina host. Todo el proyecto se compila y ejecuta **dentro de contenedores Docker**.
 
---
 
## Instalación y Ejecución
 
### 1. Clonar el repositorio
 
```bash
git clone https://github.com/[organizacion]/calidad-software-2026a.git
cd calidad-software-2026a
git checkout grupo11
cd tercer-corte
```
 
### 2. Levantar todos los servicios
 
```bash
docker compose up -d
```
 
Esto levanta los siguientes contenedores:
 
| Servicio | Puerto | Descripción |
|---|---|---|
| `app` | 8081 | Aplicación Spring Boot |
| `db` | 5432 | PostgreSQL 15 |
| `sonarqube` | 9000 | Análisis de código |
| `jenkins` | 8090 | CI/CD |
 
### 3. Acceder a la aplicación
 
- **Aplicación:** http://localhost:8081
- **SonarQube:** http://localhost:9000 (admin / admin)
- **Jenkins:** http://localhost:8090
### 4. Detener los servicios
 
```bash
docker compose down
```
 
Para limpiar también los volúmenes (⚠️ borra la base de datos):
 
```bash
docker compose down -v
```
 
---
 
## Funcionalidades
 
### Gestión de Usuarios
- **RF-01.1** Registro con `username`, `email` y `password` (únicos en el sistema).
- **RF-01.2** Login con JWT (expiración 24h).
- **RF-01.3** Logout (cliente elimina token).
- **RF-01.4** Consulta del perfil propio.
### Gestión de Proyectos
- **RF-02.1** Crear proyecto (queda como owner).
- **RF-02.2** Listar proyectos donde se es owner o miembro.
- **RF-02.3** Ver detalle con tareas agrupadas por estado.
- **RF-02.4** Editar (solo owner).
- **RF-02.5** Eliminar con cascada (solo owner).
- **RF-02.6** Invitar miembro por username (solo owner).
- **RF-02.7** Remover miembro — desasigna automáticamente sus tareas (solo owner).
- **RF-02.8** Listar miembros (owner y miembros).
### Gestión de Tareas
- **RF-03.1** Crear tarea (owner o miembros) → estado inicial `PENDING`.
- **RF-03.2** Editar tarea (owner o miembros).
- **RF-03.3** Eliminar tarea (solo owner).
- **RF-03.4** Cambiar estado: `PENDING ↔ IN_PROGRESS ↔ DONE`.
- **RF-03.5** Asignar/desasignar tarea a miembro (solo owner).
- **RF-03.6** Listar tareas agrupadas por estado.
---
 
## Sistema de Pruebas
 
### Pruebas Unitarias (JUnit)
 
**Stack:** JUnit 5 + Mockito + JaCoCo
 
**Estado actual:** 57 tests implementados, organizados en 4 suites.
 
| Suite | Tests | Cobertura |
|---|---|---|
| `AuthenticationSuite` | `UserServiceTest` (8) + `JwtServiceTest` (5) | TU01, TU05 |
| `ProjectManagementSuite` | `ProjectServiceTest` (24) | TU02 |
| `TaskManagementSuite` | `TaskServiceTest` (20) | TU03 |
| `PersistenceSuite` | `RepositoryIntegrityTest` (pendiente) | TU04 |
 
**Cobertura JaCoCo actual:**
 
| Paquete | Instrucciones | Branches |
|---|---|---|
| `service` | **94 %** | 89 % |
| `dto.response` | 100 % | 100 % |
| `entity` | 100 % | — |
| `security` | 62 % | 0 % |
| `exception` | 22 % | 0 % |
| `controller` | 0 % | — |
| **Total** | **75 %** | **74 %** |
 
**Patrón aplicado** (Given-When-Then en español):
 
```java
// Dado: configuramos los mocks para simular la BD
when(repository.metodo()).thenReturn(valorSimulado);
 
// Cuando: llamamos al método del servicio que queremos probar
Resultado r = servicio.metodo(parametros);
 
// Entonces: verificamos que el resultado sea el esperado
assertEquals(valorEsperado, r.getCampo());
```
 
**Cómo ejecutar:**
 
```bash
# Dentro del contenedor app
docker compose exec app ./mvnw test
 
# Un archivo específico
docker compose exec app ./mvnw test -Dtest=UserServiceTest
docker compose exec app ./mvnw test -Dtest=ProjectServiceTest
docker compose exec app ./mvnw test -Dtest=TaskServiceTest
docker compose exec app ./mvnw test -Dtest=JwtServiceTest
 
# Reporte de cobertura
docker compose exec app ./mvnw verify
# Abrir: app/target/site/jacoco/index.html
```
 
---
 
### Pruebas Funcionales (Selenium)
 
**Stack:** Selenium WebDriver 4.x + Selenium Grid + Page Object Model
 
**5 casos de prueba × 2 navegadores (Chrome + Firefox) = 10 ejecuciones:**
 
| # | Caso de prueba | Flujo cubierto |
|---|---|---|
| 1 | Autenticación completa | Registrarse → iniciar sesión → cerrar sesión |
| 2 | CRUD de proyecto | Crear → editar → eliminar |
| 3 | Tablero Kanban | Crear tarea → mover Pendiente → En Progreso → Terminado |
| 4 | Colaboración | Invitar usuario → asignar tarea → remover miembro |
| 5 | Perfil | Ver datos del usuario autenticado |
 
**Patrón Page Object Model:** cada pantalla de la aplicación tiene una clase dedicada que encapsula los selectores y las acciones. Si cambia un selector, solo se modifica un archivo.
 
**Cómo ejecutar:**
 
```bash
# Levantar Selenium Grid (incluido en docker-compose.yml)
docker compose up -d selenium-hub chrome firefox
 
# Ejecutar las pruebas
cd tests/selenium
./mvnw test
 
# Reporte
# tests/selenium/target/surefire-reports/index.html
```
 
> En caso de fallos, las capturas de pantalla se guardan automáticamente en `tests/selenium/target/screenshots/`.
 
---
 
### Pruebas de Rendimiento (JMeter)
 
**Stack:** JMeter 5.x con plan único de 3 escenarios.
 
**Setup inicial:** registra 1 owner + 50 usuarios bench, crea 2 proyectos (uno con 5 tareas, otro con 100), agrega miembros y asigna tareas.
 
**Escenarios:**
 
| Escenario | Usuarios | Duración | Ramp-up | Flujo |
|---|---|---|---|---|
| **NORMAL** | 5 | 10 loops | — | login → GET /me → GET /projects → GET /small/tasks |
| **CARGA** | 30 | 60 s | 15 s | login → GET /large/tasks |
| **ESTRÉS** | 60 | 60 s | 5 s | login → GET /large/tasks |
 
**Resultados obtenidos:**
 
| Escenario | GET tasks (avg) | GET tasks (p95) | Login (avg) | Error % |
|---|---|---|---|---|
| Normal | 6 ms | 8 ms | 72 ms | 0 % |
| Carga | 137 ms | 264 ms | 178 ms | 0 % |
| Estrés | 204 ms | 540 ms | 259 ms | 36,5 % |
 
**Cuellos de botella identificados:**
 
1. **N+1 queries en `GET /tasks`** — degradación de 23× entre Normal y Carga. Las relaciones `LAZY` `createdBy` y `assignedTo` disparan ~201 queries por petición.
2. **Saturación bajo estrés** — 36,5 % de errores divididos 50/50 entre 401 (login) y 403 (GET con token inválido). El pool de HikariCP (10 por defecto) + BCrypt no sostienen 60 usuarios concurrentes.
**Refactor propuesto:**
- Aplicar `JOIN FETCH` en `TaskRepository`.
- Incrementar `spring.datasource.hikari.maximum-pool-size=30`.
**Cómo ejecutar:**
 
```bash
cd tests/jmeter
jmeter -n -t plan-pruebas.jmx -l results.jtl -e -o reports/
# Abrir: reports/index.html
```
 
---
 
### Análisis de Calidad (SonarQube)
 
**Stack:** SonarQube Community Edition (en Docker, puerto 9000).
 
**Métricas analizadas:**
- Cobertura de líneas (importada desde JaCoCo).
- Code smells, bugs y vulnerabilidades.
- Duplicación de código.
- Rating de seguridad y mantenibilidad (A–E).
- *Defect density* (defectos por KLOC).
**Configuración inicial:**
 
1. Acceder a http://localhost:9000 (admin / admin).
2. Generar un token en *My Account → Security*.
3. Configurar en `app/pom.xml` o pasar por línea de comandos.
**Cómo ejecutar:**
 
```bash
docker compose exec app ./mvnw verify sonar:sonar \
  -Dsonar.host.url=http://sonarqube:9000 \
  -Dsonar.login=<TOKEN>
```
 
El reporte queda disponible en el dashboard de SonarQube.
 
---
 
## Integración Continua (Jenkins)
 
Jenkins corre en `http://localhost:8090` dentro del docker-compose. La imagen está personalizada con un `Dockerfile` propio que incluye Java 21 (la imagen oficial no la trae).
 
**Pipeline declarativo (`Jenkinsfile`):**
 
```
Checkout → Compile → Unit Tests → JaCoCo Report
         → Selenium Tests → JMeter Tests → SonarQube Scan
         → Publicar Reportes → Notificar
```
 
**Configuración inicial:**
 
1. Acceder a `http://localhost:8090`.
2. Desbloquear con la contraseña inicial:
   ```bash
   docker compose exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
   ```
3. Instalar plugins recomendados + *Pipeline*, *JaCoCo*, *HTML Publisher*, *SonarQube Scanner*.
4. Crear un Pipeline apuntando al `Jenkinsfile` del repositorio.
> Alternativa CI/CD: el repositorio incluye también workflows de **GitHub Actions** en `.github/workflows/` que ejecutan el mismo pipeline en cada push.
 
---
 
## Endpoints de la API
 
Todos los endpoints (excepto `/api/auth/**`) requieren el header:
 
```
Authorization: Bearer <token>
```
 
### Autenticación
| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/api/auth/register` | Crear cuenta |
| `POST` | `/api/auth/login` | Iniciar sesión (devuelve JWT) |
| `GET` | `/api/auth/me` | Perfil del usuario autenticado |
 
### Proyectos
| Método | Endpoint | Permisos |
|---|---|---|
| `POST` | `/api/projects` | Autenticado |
| `GET` | `/api/projects` | Owner o miembro |
| `GET` | `/api/projects/{id}` | Owner o miembro |
| `PUT` | `/api/projects/{id}` | Solo owner |
| `DELETE` | `/api/projects/{id}` | Solo owner |
| `POST` | `/api/projects/{id}/members` | Solo owner |
| `GET` | `/api/projects/{id}/members` | Owner o miembro |
| `DELETE` | `/api/projects/{id}/members/{userId}` | Solo owner |
 
### Tareas
| Método | Endpoint | Permisos |
|---|---|---|
| `POST` | `/api/projects/{projectId}/tasks` | Owner o miembro |
| `GET` | `/api/projects/{projectId}/tasks` | Owner o miembro |
| `PUT` | `/api/projects/{projectId}/tasks/{taskId}` | Owner o miembro |
| `DELETE` | `/api/projects/{projectId}/tasks/{taskId}` | Solo owner |
| `PATCH` | `/api/projects/{projectId}/tasks/{taskId}/status` | Owner o asignado |
| `PATCH` | `/api/projects/{projectId}/tasks/{taskId}/assign` | Solo owner |
 
---
 
## Decisiones de Diseño
 
| Decisión | Justificación |
|---|---|
| Sin campo `role` en `User` | Los roles se manejan a nivel de proyecto vía `ProjectMember` (owner / miembro). Esto simplifica el modelo. |
| JWT *stateless* sin invalidación server-side | El token expira a las 24h. No se mantiene blacklist; el logout es solo del lado del cliente. |
| Cascadas a nivel BD con `@OnDelete(CASCADE)` | Evita problemas de cascada-fantasma de JPA y deja la integridad referencial en la BD (más confiable). |
| `User` como principal del `SecurityContext` | Evita la indirección de `UserDetails` cuando ya tenemos la entidad. |
| Sin `UserDetailsServiceImpl` | El `JwtAuthenticationFilter` resuelve directamente contra `UserRepository`. |
| Excepciones de negocio genéricas | `ConflictException`, `ForbiddenException`, `UnauthorizedException`, `ResourceNotFoundException` → menos clases que mantener, manejadas en `GlobalExceptionHandler`. |
 
---
 
## Autores
 
**Calidad de Software 2026A**
 
- Elkin Stiven Contreras Rojas 
- David Felipe Perdomo Castillo
- David Santiago Gomez

**Docente:** Jose Miguel Mosquera
 
**Institución:** Corporación Universitaria del Huila — CORHUILA
 
---
 
## Licencia
 
Proyecto académico desarrollado en el marco de la asignatura *Calidad de Software 2026A*. Uso educativo.
