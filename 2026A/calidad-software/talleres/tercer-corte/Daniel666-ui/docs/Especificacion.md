# Especificación del Proyecto — Task Manager
**Materia:** Calidad de Software 2026A  
**Proyecto:** Proyecto Integrador — Task Manager  
**Tecnologías:** Java 11, Spring Boot 2.x, Maven, H2 (in-memory), Docker Compose  
**Versión:** 1.0.0

## 1. Objetivo
Construir una aplicación web de gestión de tareas con API REST y controles de seguridad, asegurando calidad mediante pruebas automatizadas (unitarias, integración, UI y rendimiento) y análisis estático con SonarQube, ejecutable en un entorno reproducible con Docker.

## 2. Alcance
Incluye:
- UI web (Thymeleaf) con autenticación y operaciones CRUD de tareas.
- API REST para tareas y usuarios.
- Seguridad (roles USER/ADMIN) para UI y API.
- Persistencia en H2 embebida en memoria y datos semilla.
- Infraestructura con Docker Compose y reportes en `reports/`.

No incluye:
- Persistencia en base de datos externa (se usa H2 embebida).
- Gestión avanzada de permisos por recurso (solo roles globales).

## 3. Usuarios y roles
- **ADMIN**
  - Acceso completo UI.
  - Acceso a endpoints administrativos de usuarios en API.
- **USER**
  - Acceso a UI.
  - Acceso a API de tareas (según reglas configuradas).

Credenciales de prueba:
- `admin/admin` (ADMIN)
- `user/user` (USER)

## 4. Requisitos funcionales (RF)
- **RF-01 Autenticación:** el sistema permite iniciar sesión mediante formulario (`/login`).
- **RF-02 Dashboard:** tras autenticación, el usuario visualiza un dashboard (`/dashboard`).
- **RF-03 Listar tareas:** el usuario puede ver todas las tareas (`/tasks`).
- **RF-04 Crear tarea:** el usuario puede crear tareas (`/tasks/new`).
- **RF-05 Editar tarea:** el usuario puede editar una tarea existente (`/tasks/{id}/edit`).
- **RF-06 Eliminar tarea:** el usuario puede eliminar tareas desde la lista.
- **RF-07 Filtrar tareas:** el usuario puede filtrar por estado (PENDING, IN_PROGRESS, COMPLETED).
- **RF-08 API Tareas:** el sistema expone endpoints REST para listar/crear/consultar/actualizar/eliminar tareas (`/api/tasks/**`).
- **RF-09 API Usuarios:** el sistema expone registro público y administración restringida de usuarios (`/api/users/**`).
- **RF-10 Manejo de errores:** el sistema devuelve páginas de error en UI y JSON estándar en API.

## 5. Requisitos no funcionales (RNF)
- **RNF-01 Reproducibilidad:** ejecución del stack con Docker Compose.
- **RNF-02 Seguridad:** autenticación y autorización con Spring Security; API requiere HTTP Basic.
- **RNF-03 Calidad de código:** análisis estático y Quality Gate en SonarQube.
- **RNF-04 Cobertura:** generación de reporte de cobertura (JaCoCo) para el análisis en SonarQube.
- **RNF-05 Rendimiento:** pruebas de carga con JMeter; reportes HTML almacenados en `reports/`.
- **RNF-06 Trazabilidad:** reportes de ejecución de Selenium (ExtentReports) y artefactos en `reports/`.

## 6. Arquitectura (visión técnica)
Arquitectura por capas:
- **web (MVC)**: controladores Thymeleaf, navegación y formularios.
- **api (REST)**: controladores REST para integración y pruebas.
- **service**: lógica de negocio y validaciones.
- **repository**: persistencia (Spring Data JPA).
- **common**: excepciones y handlers.
- **security**: configuración Spring Security y UserDetailsService.

## 7. Interfaces
### 7.1 UI (Thymeleaf)
- Login: `/login`
- Dashboard: `/dashboard`
- Tareas: `/tasks` (crear/editar/eliminar/filtrar)

### 7.2 API REST (HTTP Basic)
Tareas:
- `GET /api/tasks`
- `GET /api/tasks?status=PENDING`
- `POST /api/tasks`
- `GET /api/tasks/{id}`
- `PUT /api/tasks/{id}`
- `DELETE /api/tasks/{id}`

Usuarios:
- `POST /api/users/register` (público)
- `GET /api/users` (solo ADMIN)
- `GET /api/users/{id}` (solo ADMIN)
- `DELETE /api/users/{id}` (solo ADMIN)

## 8. Datos y persistencia
- Base de datos: **H2 embebida in-memory**.
- Datos semilla: `data.sql` (usuarios y tareas iniciales).

## 9. Supuestos y limitaciones
- Las pruebas UI pueden ejecutarse contra Selenium Grid o un contenedor standalone Chrome; la descarga de imágenes puede depender de la red disponible.
