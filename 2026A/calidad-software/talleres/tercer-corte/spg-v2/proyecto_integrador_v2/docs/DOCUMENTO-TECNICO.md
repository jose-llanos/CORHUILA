# SGP LAB - Documento Tecnico

**Sistema de Gestion de Prestamos de Laboratorio**
Backend: Java 17 + Spring Boot 3.3.5 | Frontend: Vue 3 + Vuetify | Calidad: JaCoCo + JMeter + SonarQube + Jenkins

---

## Tabla de contenidos

1. Vision general
2. Arquitectura y decisiones tecnicas
3. Modelo de dominio
4. Seguridad: SHA-256 + JWT
5. Estructura de paquetes
6. Como ejecutar el proyecto
7. Calidad de codigo con JaCoCo
8. Pruebas de performance con JMeter
9. Automatizacion con Jenkins
10. Documentacion con Javadoc
11. Analisis estatico con SonarQube
12. Apendice A: Endpoints disponibles
13. Apendice B: Decisiones de seguridad

---

## 1. Vision general

SGP LAB administra inventario y prestamos de equipos de laboratorio. Maneja dos roles: **ADMINISTRADOR** (control total) y **CLIENTE** (consulta y solicitud de prestamos).

Casos de uso implementados:

- Autenticacion con email + contrasena (cifrada SHA-256 con salt).
- Sesion mediante JWT (HS256) con expiracion de 1 hora.
- CRUD de Usuarios, Equipos, Prestamos y Penalizaciones.
- Reglas de negocio: un usuario no puede tener mas de un prestamo ACTIVO; el stock se decrementa al crear el prestamo y se restituye al marcarlo DEVUELTO; un equipo con prestamos activos no se puede eliminar.

---

## 2. Arquitectura y decisiones tecnicas

### Stack

| Capa            | Tecnologia                          |
|-----------------|-------------------------------------|
| Backend         | Java 17, Spring Boot 3.3.5          |
| Persistencia    | Spring Data JPA + Hibernate         |
| Base de datos   | PostgreSQL (dev/prod) + H2 (tests)  |
| Seguridad       | Spring Security + JWT (jjwt 0.12.6) |
| Cifrado pwd     | SHA-256 + salt (MessageDigest)      |
| Frontend        | Vue 3 + Vite + Vuetify 3 + Axios    |
| Build           | Maven 3.9+                          |
| Cobertura       | JaCoCo 0.8.12 (umbral 80%/70%)      |
| Performance     | Apache JMeter 5.6.3                 |
| Analisis        | SonarQube                           |
| CI              | Jenkins (Jenkinsfile declarativo)   |
| Docs            | Maven Javadoc Plugin                |

### Capas (Clean Architecture-lite)

```
Controller  -> recibe HTTP, valida @Valid, delega
   |
Service     -> reglas de negocio, transacciones, logging
   |
Repository  -> Spring Data JPA, acceso a BD
   |
Entity      -> JPA, mapeo a tablas
```

DTOs separados de entidades: **nunca** se exponen entidades JPA en respuestas (evita filtrar `passwordHash`, lazy proxies, etc).

### Decisiones tecnicas clave

1. **Java 17 (no 21)**: maxima compatibilidad con Jenkins y entornos corporativos.
2. **Spring Boot 3.3.5 estable**: el original usaba `3.5.15-SNAPSHOT` que no esta liberado.
3. **DTOs planos**: `usuarioId` en lugar de `usuario: { id: X }`. Reduce acoplamiento entre cliente y modelo interno.
4. **GlobalExceptionHandler** con respuestas uniformes (`ErrorResponse { timestamp, status, error, message, path, fieldErrors }`).
5. **Perfiles de configuracion**: `dev` (PostgreSQL local), `test` (H2 en memoria). Los tests jamas tocan tu BD real.
6. **JWT en header `Authorization: Bearer <token>`**: stateless, sin sesiones en servidor, ideal para escalado horizontal.

---

## 3. Modelo de dominio

### Entidades

```
Usuario (id, nombre, email UNIQUE, passwordHash, estado, rol)
Equipo  (id, codigoInventario UNIQUE, nombre, cantidad, estado)
Prestamo (id, fechaInicio, fechaFin, equipo FK, usuario FK, estado)
Penalizacion (id, motivo, fechaInicio, fechaFin, usuario FK, estado)
```

### Enums

- `Rol`: `ADMINISTRADOR`, `CLIENTE`
- `EstadoUsuario`: `ACTIVO`, `PENALIZADO`
- `EstadoEquipo`: `DISPONIBLE`, `PRESTADO`, `EN_MANTENIMIENTO`
- `EstadoPrestamo`: `ACTIVO`, `DEVUELTO`, `VENCIDO`, `CANCELADO`
- `EstadoPenalizacion`: `ACTIVA`, `CUMPLIDA`, `LEVANTADA`

---

## 4. Seguridad: SHA-256 + JWT

### 4.1 Cifrado de contrasenas

Implementado en `PasswordHashUtil`:

- Salt de 16 bytes aleatorios (SecureRandom) por contrasena.
- Hash: `SHA-256(salt || password)`.
- Formato persistido: `saltHex:hashHex` (un solo campo VARCHAR(200)).
- Verificacion con `MessageDigest.isEqual` (constante en tiempo, resistente a timing attacks).

```java
String stored = PasswordHashUtil.hash("password");
// "a1b2...d6:6caa...d2"
boolean ok = PasswordHashUtil.matches("password", stored);  // true
```

**Nota academica**: SHA-256 es rapido (200M/seg en GPU moderna). Para produccion en sistemas con superficie de ataque externa se recomienda **BCrypt** o **Argon2** (cost-factor adaptativo). El uso de SHA-256 en este proyecto responde a un requisito academico explicito.

### 4.2 Flujo de autenticacion JWT

```
1. Cliente envia POST /api/auth/login con email + password.
2. AuthService busca usuario por email.
3. PasswordHashUtil.matches() verifica contra el hash almacenado.
4. Si OK, JwtService.generateToken() emite un JWT HS256:
     header  { alg: "HS256", typ: "JWT" }
     payload { sub: email, iss: "sgp-lab", uid: id, rol: "ADMINISTRADOR|CLIENTE",
               nombre: "...", iat, exp }
5. Cliente guarda el token en localStorage (frontend Vue).
6. En cada request, el frontend envia: Authorization: Bearer <token>
7. JwtAuthenticationFilter intercepta, valida firma + expiracion,
   y autentica al usuario en el contexto de Spring Security.
8. @PreAuthorize("hasRole('ADMINISTRADOR')") protege endpoints sensibles.
```

### 4.3 Endpoints publicos vs protegidos

- Publicos: `POST /api/auth/login`, `GET /actuator/health`, `OPTIONS /**` (CORS preflight).
- Protegidos: el resto requiere JWT valido.
- Solo admin: `GET/POST/PUT/DELETE /api/usuarios`, `POST/PUT/DELETE /api/equipos`, `GET/POST/PUT/DELETE /api/penalizaciones`.

---

## 5. Estructura de paquetes

```
com.sgplab.backend
├── SgpBackendApplication.java
├── config/
│   └── SecurityConfig.java                # CORS, filter chain, @PreAuthorize habilitado
├── controller/                            # REST controllers (5)
│   ├── AuthController.java
│   ├── UsuarioController.java
│   ├── EquipoController.java
│   ├── PrestamoController.java
│   └── PenalizacionController.java
├── dto/
│   ├── request/                           # Inputs validados con @Valid
│   ├── response/                          # Outputs (nunca exponen passwordHash)
│   └── mapper/                            # Entity <-> DTO
├── exception/
│   ├── GlobalExceptionHandler.java        # @RestControllerAdvice
│   ├── ErrorResponse.java                 # Estructura uniforme
│   ├── ResourceNotFoundException.java     # -> HTTP 404
│   ├── BusinessRuleException.java         # -> HTTP 409
│   ├── DuplicateResourceException.java    # -> HTTP 409
│   └── InvalidCredentialsException.java   # -> HTTP 401
├── model/
│   ├── entity/  (4 entidades JPA)
│   └── enums/   (5 enums)
├── repository/  (4 repositorios JpaRepository)
├── security/
│   ├── JwtService.java
│   ├── JwtAuthenticationFilter.java
│   └── JwtAuthEntryPoint.java
├── service/
│   ├── contract/  (5 interfaces I*Service)
│   └── impl/      (5 implementaciones *ServiceImpl)
└── util/
    └── PasswordHashUtil.java
```

---

## 6. Como ejecutar el proyecto

### 6.1 Pre-requisitos

- JDK 17
- Maven 3.9+
- Docker + Docker Compose (para PostgreSQL local)
- Node.js 18+ y npm (para el frontend)

### 6.2 Arrancar PostgreSQL

```bash
cd proyectoIntegrador
docker compose up -d postgres
```

### 6.3 Backend

```bash
cd sgp-backend

# Compilar y correr
mvn spring-boot:run

# Equivalente: ejecutar el JAR generado
mvn clean package
java -jar target/sgp-backend-1.0.0.jar
```

El backend levanta en `http://localhost:8080`.

Datos iniciales (`import.sql`):

- `admin@sgplab.edu.co` / `password`  (rol ADMINISTRADOR)
- `cliente@sgplab.edu.co` / `cliente123`  (rol CLIENTE)

### 6.4 Frontend

```bash
cd sgp-frontend
npm install
npm run dev
```

El frontend abre en `http://localhost:5173`.

### 6.5 Variables de entorno (opcional, para produccion)

```bash
export SGP_JWT_SECRET="una-clave-de-al-menos-32-bytes-base64-o-utf8"
export SPRING_DATASOURCE_PASSWORD="mi-password-prod"
export SPRING_PROFILES_ACTIVE="prod"
```

---

## 7. Calidad de codigo con JaCoCo

### 7.1 Configuracion

JaCoCo esta cableado en `pom.xml` con tres ejecuciones:

1. `prepare-agent` (fase initialize): instala el agente que mide ejecucion.
2. `report` (fase test): genera HTML + XML.
3. `check-coverage` (fase verify): rompe el build si la cobertura no cumple umbrales.

Umbrales actuales:

| Metrica         | Minimo  |
|-----------------|---------|
| Cobertura lineas | 80%    |
| Cobertura branches | 70%  |

Exclusiones (paquetes que no requieren tests porque son boilerplate):

- `**/dto/**`
- `**/model/entity/**`
- `**/model/enums/**`
- `**/config/**`
- `**/SgpBackendApplication*`

### 7.2 Comandos

```bash
# Generar reporte (solo unit tests)
mvn clean test

# Cobertura + check de umbrales (incluye integracion)
mvn clean verify

# Saltar el check temporalmente (para experimentar)
mvn clean verify -Pno-coverage-check
```

### 7.3 Como leer el reporte

Abre `target/site/jacoco/index.html` en el navegador.

| Color en codigo | Significado                  |
|-----------------|------------------------------|
| Verde           | Linea cubierta               |
| Amarillo        | Branch parcialmente cubierto (algunos `if/else` no probados) |
| Rojo            | Linea no ejecutada por ningun test |

Si el build falla con:

```
Rule violated for bundle sgp-backend: lines covered ratio is 0.74, but expected minimum is 0.80
```

Significa que necesitas anadir tests al codigo que aparece rojo en el reporte.

---

## 8. Pruebas de performance con JMeter

### 8.1 Escenarios

| Plan              | Usuarios | Ramp-up | Duracion | Think time |
|-------------------|----------|---------|----------|------------|
| `carga-normal.jmx`| 50       | 30s     | 60s      | 500-1500ms |
| `estres.jmx`      | 200      | 60s     | 120s     | 100-500ms  |

Cada plan ejecuta tres samplers por iteracion:

1. `POST /api/auth/login` (con assertion de status 200 y duracion < 1000ms).
2. `GET /api/equipos` (con header `Authorization: Bearer ${jwtToken}` extraido del login).
3. `GET /api/prestamos`.

### 8.2 Ejecucion

**Modo no-GUI (recomendado, mas rapido)**:

```bash
cd sgp-backend/src/test/jmeter
./run-jmeter.sh todo   # corre ambos planes

# O por separado
./run-jmeter.sh carga
./run-jmeter.sh estres
```

**Host remoto**:

```bash
SGP_HOST=192.168.1.50 SGP_PORT=8080 ./run-jmeter.sh todo
```

**Modo GUI (solo para debugging)**:

```bash
jmeter -t carga-normal.jmx
```

### 8.3 Reportes generados

```
sgp-backend/src/test/jmeter/
├── results/
│   ├── carga-normal.jtl
│   ├── carga-normal-summary.csv
│   ├── carga-normal-aggregate.csv
│   ├── estres.jtl
│   ├── estres-summary.csv
│   └── estres-aggregate.csv
└── reports/
    ├── carga-normal/index.html
    └── estres/index.html
```

El `index.html` de cada plan contiene un dashboard interactivo con graficas.

### 8.4 Interpretacion de metricas

Estas son las metricas que JMeter entrega y como leerlas:

| Metrica            | Que mide                                         | Buen valor (web tipico) |
|--------------------|--------------------------------------------------|-------------------------|
| **Samples**        | Numero total de requests ejecutados              | depende del plan        |
| **Average (ms)**   | Tiempo medio de respuesta                        | < 500ms                 |
| **Median (50th)**  | Tiempo de respuesta de la mitad de las requests  | < 300ms                 |
| **90% Line (p90)** | El 90% respondio en <= este valor                | < 1000ms                |
| **95% Line (p95)** | El 95% respondio en <= este valor                | < 1500ms                |
| **99% Line (p99)** | El 99% respondio en <= este valor                | < 3000ms                |
| **Min / Max**      | Mejor y peor caso (sirve para detectar outliers) | -                       |
| **Throughput**     | Requests por segundo (TPS)                       | depende; mayor es mejor |
| **Received KB/sec**| Ancho de banda de salida                         | -                       |
| **Sent KB/sec**    | Ancho de banda de entrada                        | -                       |
| **Error %**        | Porcentaje de requests con status != 2xx         | < 1%                    |

#### Tabla de resultados esperados (referencia)

Para una maquina local de desarrollo (8GB RAM, SSD, Postgres en docker):

| Escenario        | TPS aprox | p95 esperado | p99 esperado | Error % esperado |
|------------------|-----------|--------------|--------------|------------------|
| Carga normal 50  | 150-200   | < 400ms      | < 800ms      | 0%               |
| Estres 200       | 300-400   | < 800ms      | < 2000ms     | < 1%             |

Si los resultados son **muy** peores:

- p95 disparado -> hay un endpoint lento (probablemente `POST /prestamos`, que actualiza dos tablas).
- TPS bajo y CPU del backend al 100% -> bottleneck en CPU (tipicamente el hash SHA-256 del login, que es deliberadamente lento).
- TPS bajo y CPU bajo -> bottleneck en BD; revisa pool de conexiones (`spring.datasource.hikari.maximum-pool-size`).

### 8.5 Customizar el plan

Para anadir mas usuarios al CSV de credenciales:

```bash
# Edita sgp-backend/src/test/jmeter/data/usuarios.csv
echo "user1@x.com,pwd123" >> usuarios.csv
```

Para cambiar la duracion sin editar el `.jmx`, usa propiedades:

```bash
jmeter -n -t carga-normal.jmx -Jhost=localhost -Jport=8080 -l out.jtl
```

---

## 9. Automatizacion con Jenkins

### 9.1 Configuracion del job

1. Crea un job tipo **Multibranch Pipeline** (o **Pipeline** simple).
2. Apunta al repositorio Git.
3. Script Path: `Jenkinsfile` (en la raiz del proyecto).
4. Configura herramientas globales en Jenkins:
   - JDK con name `jdk17` apuntando a Java 17.
   - Maven con name `maven-3.9` apuntando a Maven 3.9+.
5. (Opcional) Credenciales:
   - `sonar-token`: token de SonarQube tipo "Secret Text".

### 9.2 Stages del pipeline

| # | Stage                | Que hace                                            | Falla el build? |
|---|----------------------|-----------------------------------------------------|-----------------|
| 1 | Checkout             | Descarga el codigo                                  | Si             |
| 2 | Build                | `mvn clean compile`                                 | Si             |
| 3 | Tests unitarios      | `mvn test` + publica JUnit                          | Si             |
| 4 | Tests integracion    | `mvn verify` + publica JUnit + jacoco:check         | Si (umbral)    |
| 5 | Cobertura JaCoCo     | Publica HTML                                        | No             |
| 6 | Javadoc              | `mvn javadoc:javadoc` + publica HTML                | No             |
| 7 | SonarQube            | `mvn sonar:sonar` (solo si SONAR_HOST_URL definido) | No             |
| 8 | JMeter remoto        | Corre `run-jmeter.sh todo` (solo si RUN_PERFORMANCE=true) | No        |
| 9 | Empaquetar           | `mvn package` + archiva el JAR                      | Si             |

### 9.3 Ejecutar JMeter remoto

En el job de Jenkins, configurar parametros:

- `RUN_PERFORMANCE` = `true`
- `REMOTE_TEST_HOST` = `192.168.1.50` (donde corre el backend bajo prueba)
- `REMOTE_TEST_PORT` = `8080`

El stage 8 ejecutara JMeter contra ese host y publicara los reportes HTML como artefactos del build.

### 9.4 Como leer los resultados en Jenkins

Despues de un build exitoso, la sidebar del build incluye:

- **JaCoCo Coverage Report**: HTML navegable con cobertura por clase.
- **Javadoc**: documentacion generada.
- **JMeter Carga Normal** y **JMeter Estres**: dashboards de performance.
- **Artefactos**: el JAR final + los CSVs de JMeter.

---

## 10. Documentacion con Javadoc

### 10.1 Generar desde la linea de comandos

```bash
cd sgp-backend

# Generar reporte HTML (en target/reports/apidocs)
mvn javadoc:javadoc

# Generar JAR de documentacion adjuntable a Nexus
mvn javadoc:jar
```

### 10.2 Generar desde IntelliJ IDEA

**Opcion A - Menu Tools (IntelliJ)**:

1. `Tools` -> `Generate JavaDoc...`
2. Output directory: `<proyecto>/sgp-backend/target/javadoc-idea`
3. Other command line arguments: `-encoding UTF-8 -charset UTF-8`
4. Click OK.

**Opcion B - Goal de Maven desde IntelliJ**:

1. Abrir el panel **Maven** (lateral derecho).
2. Expandir `sgp-backend` -> `Plugins` -> `javadoc`.
3. Doble click en `javadoc:javadoc`.
4. Cuando termina, el output esta en `target/reports/apidocs/index.html`.
5. Click derecho sobre `index.html` -> `Open in Browser`.

**Opcion C - Run Configuration personalizada**:

1. `Run` -> `Edit Configurations...` -> `+` -> `Maven`.
2. Name: `Generate Javadoc`.
3. Command line: `javadoc:javadoc`.
4. Working directory: `$ProjectFileDir$/sgp-backend`.
5. Apply.

### 10.3 Configuracion del plugin

Definida en `pom.xml`:

```xml
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-javadoc-plugin</artifactId>
    <version>3.10.1</version>
    <configuration>
        <doclint>none</doclint>
        <encoding>UTF-8</encoding>
        <show>protected</show>
        <author>true</author>
        <windowtitle>SGP LAB Backend API ${project.version}</windowtitle>
        <doctitle>SGP LAB - Sistema de Gestion de Prestamos de Laboratorio</doctitle>
    </configuration>
</plugin>
```

`doclint=none` evita que warnings de Javadoc rompan el build.
`show=protected` incluye miembros publicos y protegidos (omite los privados).

Todas las clases publicas estan documentadas con tags `@author`, `@version`, `@see`, `@param`, `@return`, `@throws`.

---

## 11. Analisis estatico con SonarQube

### 11.1 Levantar SonarQube local

```bash
docker compose up -d sonarqube
# Espera ~1 minuto y abre http://localhost:9000
# Credenciales iniciales: admin/admin (te pedira cambiar)
```

### 11.2 Generar token de analisis

1. En SonarQube: `My Account` -> `Security` -> Generate Token.
2. Cuando uses Jenkins, agrega ese token como credencial `sonar-token`.

### 11.3 Ejecutar analisis

```bash
cd sgp-backend
mvn clean verify sonar:sonar \
  -Dsonar.host.url=http://localhost:9000 \
  -Dsonar.login=<tu-token>
```

Resultado: proyecto `sgp-lab-backend` visible en SonarQube con:

- Bugs, vulnerabilidades, code smells.
- Cobertura (lee el XML de JaCoCo que se genera en `verify`).
- Duplicated lines, complejidad ciclomatica, deuda tecnica.

### 11.4 Quality Gate sugerido

| Condicion                  | Umbral   |
|----------------------------|----------|
| Coverage on new code       | >= 80%   |
| Duplicated lines on new code | <= 3%  |
| Maintainability rating     | A        |
| Reliability rating         | A        |
| Security rating            | A        |

---

## 12. Apendice A: Endpoints disponibles

| Metodo | Path                                  | Rol requerido  | Descripcion                                      |
|--------|---------------------------------------|----------------|--------------------------------------------------|
| POST   | `/api/auth/login`                     | -              | Login. Devuelve JWT.                             |
| GET    | `/api/usuarios`                       | ADMINISTRADOR  | Lista todos los usuarios.                        |
| GET    | `/api/usuarios/{id}`                  | ADMIN o CLIENTE | Obtiene un usuario por id.                      |
| POST   | `/api/usuarios`                       | ADMINISTRADOR  | Crea un usuario (password obligatoria).          |
| PUT    | `/api/usuarios/{id}`                  | ADMINISTRADOR  | Actualiza (password opcional).                   |
| DELETE | `/api/usuarios/{id}`                  | ADMINISTRADOR  | Elimina un usuario.                              |
| GET    | `/api/equipos`                        | autenticado    | Lista equipos.                                   |
| GET    | `/api/equipos/{id}`                   | autenticado    | Obtiene un equipo.                               |
| POST   | `/api/equipos`                        | ADMINISTRADOR  | Crea un equipo.                                  |
| PUT    | `/api/equipos/{id}`                   | ADMINISTRADOR  | Actualiza un equipo.                             |
| DELETE | `/api/equipos/{id}`                   | ADMINISTRADOR  | Elimina (falla si tiene prestamos activos).      |
| GET    | `/api/prestamos`                      | autenticado    | Lista prestamos.                                 |
| GET    | `/api/prestamos/{id}`                 | autenticado    | Obtiene un prestamo.                             |
| POST   | `/api/prestamos`                      | autenticado    | Crea prestamo (descuenta stock).                 |
| PUT    | `/api/prestamos/{id}`                 | ADMINISTRADOR  | Actualiza estado (DEVUELTO restituye stock).     |
| DELETE | `/api/prestamos/{id}`                 | ADMINISTRADOR  | Elimina (falla si ACTIVO).                       |
| GET    | `/api/penalizaciones`                 | ADMINISTRADOR  | Lista todas.                                     |
| GET    | `/api/penalizaciones/{id}`            | autenticado    | Obtiene una.                                     |
| GET    | `/api/penalizaciones/usuario/{id}/activa` | autenticado | True/false si usuario esta penalizado activamente. |
| POST   | `/api/penalizaciones`                 | ADMINISTRADOR  | Crea penalizacion.                               |
| PUT    | `/api/penalizaciones/{id}`            | ADMINISTRADOR  | Actualiza.                                       |
| DELETE | `/api/penalizaciones/{id}`            | ADMINISTRADOR  | Elimina.                                         |

Todos los endpoints (excepto `/api/auth/login`) requieren header:

```
Authorization: Bearer <jwt>
```

---

## 13. Apendice B: Decisiones de seguridad

| Decision                                                   | Razon                                                  |
|------------------------------------------------------------|--------------------------------------------------------|
| SHA-256 con salt por usuario                               | Requisito academico; salt unico previene rainbow tables |
| JWT en lugar de sesion HTTP                                | Stateless, escalable, sin afinidad de sesion           |
| Expiracion del token: 1h                                   | Balance entre UX y limitar danio si se roba el token   |
| `MessageDigest.isEqual` en `matches()`                     | Resistente a timing attacks                            |
| CSRF deshabilitado                                         | API REST stateless con JWT; CSRF aplica a sesiones cookie |
| CORS limitado a `localhost:5173/5174/4173`                 | Solo el frontend autorizado puede consumir el API      |
| `H2 console` permitida solo si esta en el path             | Solo accesible en perfil dev                           |
| `passwordHash` excluido de DTOs de respuesta               | El frontend nunca recibe el hash                       |
| Mensaje generico "Credenciales invalidas" en login fallido | Evita enumerar emails registrados                      |
| Validacion `@Valid` en todos los DTOs de entrada           | Defense in depth contra payloads malformados           |
| `spring-boot-devtools` solo en runtime                     | No se incluye en el JAR final de produccion            |
| `application.yml` sin secretos en codigo                   | El JWT secret y la password de Postgres se inyectan por variables de entorno en prod |

---

## Anexo - Como probar end-to-end manualmente

1. Arranca PostgreSQL: `docker compose up -d postgres`
2. Arranca backend: `mvn spring-boot:run` (en `sgp-backend/`)
3. Arranca frontend: `npm run dev` (en `sgp-frontend/`)
4. Abre `http://localhost:5173`
5. Login con `admin@sgplab.edu.co` / `password`
6. Crea un nuevo usuario CLIENTE con su password.
7. Logout. Login de nuevo con el nuevo cliente.
8. Verifica que solo ves tu vista de cliente (no `/usuarios`, `/penalizaciones`).
9. Solicita un prestamo desde el catalogo.
10. Vuelve a hacer login como admin y aprueba/cambia el estado.

Alternativa con Postman: importa `postman/SGP-LAB.postman_collection.json` y ejecuta los requests en orden empezando por **Auth > Login (admin)**.

---

*SGP LAB v1.0.0 - 2026*
