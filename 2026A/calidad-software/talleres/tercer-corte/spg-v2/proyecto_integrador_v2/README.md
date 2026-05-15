# SGP LAB - Sistema de Gestion de Prestamos de Laboratorio

Backend Spring Boot 3.3.5 (Java 17) + Frontend Vue 3 + Vuetify, con automatizacion de calidad de software integrada.

## Opciones de ejecucion

### Opcion A - Todo dockerizado (recomendado para evaluadores)

```bash
# 1. Configurar variables de entorno
cp .env.example .env
# (opcional: ajustar SGP_JWT_SECRET, passwords, etc.)

# 2. Levantar la app completa: db + backend + frontend
docker compose up -d --build

# 3. Acceder
#    Frontend: http://localhost:5173
#    Backend:  http://localhost:8080
#    Postgres: localhost:5433 (admin/password123)
```

Incluir SonarQube en la subida:

```bash
docker compose --profile quality up -d
# SonarQube: http://localhost:9000  (admin/admin)
```

Incluir Jenkins:

```bash
docker compose --profile ci up -d
# Jenkins: http://localhost:8090
```

Todo a la vez:

```bash
docker compose --profile quality --profile ci --profile performance up -d --build
```

Bajar y borrar volumenes:

```bash
docker compose down -v
```

### Opcion B - Solo PostgreSQL en Docker, backend y front locales (recomendado para desarrollo)

```bash
# 1. PostgreSQL en docker
docker compose up -d postgres

# 2. Backend
cd sgp-backend
mvn spring-boot:run
# -> http://localhost:8080

# 3. Frontend (otra terminal)
cd sgp-frontend
npm install
npm run dev
# -> http://localhost:5173
```

## Credenciales precargadas

| Usuario                  | Password    | Rol           |
|--------------------------|-------------|---------------|
| `admin@sgplab.edu.co`    | `password`  | ADMINISTRADOR |
| `cliente@sgplab.edu.co`  | `cliente123`| CLIENTE       |

## Comandos clave

```bash
# Tests + cobertura JaCoCo (umbral 80% lineas / 70% branches)
cd sgp-backend
mvn clean verify

# Javadoc
mvn javadoc:javadoc
# Abrir target/reports/apidocs/index.html

# Tests de performance JMeter (con backend corriendo)
cd src/test/jmeter
./run-jmeter.sh todo

# Analisis SonarQube
mvn sonar:sonar \
  -Dsonar.host.url=http://localhost:9000 \
  -Dsonar.login=<tu-token>
```

## Puertos por defecto

| Servicio    | Puerto host | Puerto contenedor |
|-------------|-------------|-------------------|
| Frontend    | 5173        | 80                |
| Backend     | 8080        | 8080              |
| PostgreSQL  | 5433        | 5432              |
| SonarQube   | 9000        | 9000              |
| Jenkins     | 8090        | 8080              |

## Estructura

```
proyectoIntegrador/
├── README.md
├── .env.example                # Variables del docker-compose
├── .gitignore                  # Ignora target, node_modules, etc.
├── Jenkinsfile                 # CI/CD pipeline declarativo
├── docker-compose.yml          # Orquesta toda la stack
├── docs/
│   └── DOCUMENTO-TECNICO.md
├── postman/
│   └── SGP-LAB.postman_collection.json
├── sgp-backend/                # Spring Boot
│   ├── Dockerfile              # Multi-stage Maven + JRE 17
│   ├── .dockerignore
│   ├── pom.xml
│   └── src/
└── sgp-frontend/               # Vue 3 + Vuetify
    ├── Dockerfile              # Multi-stage Node + Nginx
    ├── .dockerignore
    ├── nginx.conf              # Proxy /api -> backend + SPA fallback
    ├── package.json
    └── src/
```

## Documentacion

- **Documento tecnico completo**: [`docs/DOCUMENTO-TECNICO.md`](docs/DOCUMENTO-TECNICO.md)
- **Coleccion Postman**: [`postman/SGP-LAB.postman_collection.json`](postman/SGP-LAB.postman_collection.json)
- **Pipeline Jenkins**: [`Jenkinsfile`](Jenkinsfile)

## Licencia

Proyecto academico - Universidad CORHUILA, 2026A - Calidad de Software.
