# Proyecto Integrador — Calidad de Software 2026A
## Task Manager (Spring Boot + H2 + QA)

Sistema de Gestión de Tareas (Task Manager) desarrollado en **Java 11 + Spring Boot 2.x** con:
- UI web con **Thymeleaf** (login, dashboard, lista y formulario de tareas).
- API REST para integración de pruebas: **/api/tasks** y **/api/users**.
- Autenticación y autorización con **Spring Security** (roles **USER** y **ADMIN**).
- Base de datos **H2** en memoria (embebida, para simplicidad y reproducibilidad).
- SonarQube incluido en Docker Compose.

---

## Requisitos previos
- Docker Desktop (Windows/Mac/Linux)
- Docker Compose (incluido en Docker Desktop)

---

## Ejecución con Docker (un solo comando)
1) Ubícate en la carpeta `docker/`:
```bash
cd docker
```

2) Levanta todo:
```bash
docker compose up --build
```

### URLs de acceso
- Aplicación: http://localhost:8080
- SonarQube: http://localhost:9000
- Selenium Grid: http://localhost:4444
- H2 Console: http://localhost:8080/h2-console

---

## Credenciales de prueba
- ADMIN: **admin / admin**
- USER: **user / user**

---

## Uso funcional (UI)
- Login: `/login`
- Dashboard: `/dashboard`
- Tareas: `/tasks`
  - Crear: `/tasks/new`
  - Editar: `/tasks/{id}/edit`
  - Eliminar: botón Delete en la lista
  - Filtrar por estado: selector en la lista (PENDING/IN_PROGRESS/COMPLETED)

---

## API REST (para integración de pruebas)
La API requiere autenticación **HTTP Basic**.

### Tasks
- `GET /api/tasks`
- `GET /api/tasks?status=PENDING`
- `POST /api/tasks`
- `GET /api/tasks/{id}`
- `PUT /api/tasks/{id}`
- `DELETE /api/tasks/{id}`

Ejemplo (PowerShell) creando una tarea:
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8080/api/tasks `
  -Authentication Basic -Credential (Get-Credential) `
  -ContentType "application/json" `
  -Body '{"title":"New task","description":"Created from API","status":"PENDING","priority":"MEDIUM"}'
```

### Users
- `POST /api/users/register` (público)
- `GET /api/users` (solo ADMIN)
- `GET /api/users/{id}` (solo ADMIN)
- `DELETE /api/users/{id}` (solo ADMIN)

---

## Ejecución manual (sin Docker)
Desde la carpeta `app/`:
```bash
mvn test
mvn spring-boot:run
```

La aplicación quedará en http://localhost:8080

---

## SonarQube y Quality Gate
El análisis se envía a SonarQube (http://localhost:9000). El **Quality Gate** se interpreta así:
- **Passed/OK**: el proyecto cumple las condiciones mínimas (p. ej. sin bugs críticos, cobertura suficiente, etc.).
- **Failed/Error**: el proyecto no cumple al menos una condición del Quality Gate; se revisan los detalles para corregir.

Ejecutar análisis desde Docker (con la pila levantada):
```bash
sh docker/sonar-scan.sh
```

---

## Comandos (Makefile)
En entornos con `make` disponible:
```bash
make build
make up
make test
make selenium
make sonar
make logs
make down
```
