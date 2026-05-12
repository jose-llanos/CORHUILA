# Aplicación de Ejemplo para JMeter

Esta es una pequeña aplicación REST construida con Spring Boot para usar como objetivo en las pruebas de carga con JMeter.

## Requisitos Previos

- **Java 17 o superior**: Descargable desde [openjdk.java.net](https://openjdk.java.net/)
- **Maven 3.6.0 o superior**: Descargable desde [maven.apache.org](https://maven.apache.org/)
- **JMeter** (para ejecutar las pruebas): Ver guía principal en `index.html`

## Estructura de Directorios

```
app-ejemplo/
├── pom.xml                          # Configuración de Maven
├── src/
│   └── main/
│       ├── java/com/calidad/jmeter/
│       │   ├── ApiApplication.java          # Clase principal
│       │   └── ItemController.java          # Controlador REST
│       └── resources/
│           └── application.properties       # Configuración de Spring
└── README.md
```

## Instalación y Ejecución

### Opción 1: Compilación con Maven

```bash
# 1. Navegar al directorio de la aplicación
cd app-ejemplo

# 2. Compilar la aplicación
mvn clean package

# 3. Ejecutar la aplicación compilada
java -jar target/jmeter-app-ejemplo-1.0.0.jar
```

La aplicación estará disponible en: **http://localhost:8080**

### Opción 2: Ejecución directa con Maven

```bash
cd app-ejemplo
mvn spring-boot:run
```

## Endpoints Disponibles

### 1. **GET /api/v1/health**
Verifica que la aplicación está funcionando.

```bash
curl http://localhost:8080/api/v1/health
```

**Respuesta:**
```json
{
  "status": "UP",
  "version": "1.0.0",
  "message": "Aplicación funcionando correctamente"
}
```

---

### 2. **GET /api/v1/items**
Obtiene la lista de todos los items.

```bash
curl http://localhost:8080/api/v1/items
```

**Respuesta:**
```json
[
  {"id": 1, "nombre": "Laptop", "precio": 999.99},
  {"id": 2, "nombre": "Mouse", "precio": 25.50},
  ...
]
```

---

### 3. **GET /api/v1/items/{id}**
Obtiene un item específico por su ID.

```bash
curl http://localhost:8080/api/v1/items/1
```

**Respuesta:**
```json
{"id": 1, "nombre": "Laptop", "precio": 999.99}
```

---

### 4. **POST /api/v1/items**
Crea un nuevo item.

```bash
curl -X POST http://localhost:8080/api/v1/items \
  -H "Content-Type: application/json" \
  -d '{"nombre": "Monitor Nuevo", "precio": 350.00}'
```

---

### 5. **PUT /api/v1/items/{id}**
Actualiza un item existente.

```bash
curl -X PUT http://localhost:8080/api/v1/items/1 \
  -H "Content-Type: application/json" \
  -d '{"nombre": "Laptop Actualizada", "precio": 1199.99}'
```

---

### 6. **DELETE /api/v1/items/{id}**
Elimina un item.

```bash
curl -X DELETE http://localhost:8080/api/v1/items/1
```

---

### 7. **GET /api/v1/items/report** ⏱️
Genera un reporte (tarda 2 segundos) - **Ideal para simular operaciones lentas**.

```bash
curl http://localhost:8080/api/v1/items/report
```

## Configuración para JMeter

### Paso 1: Asegúrate que la aplicación está corriendo

```bash
curl http://localhost:8080/api/v1/health
```

### Paso 2: En JMeter, usa estas configuraciones

| Parámetro | Valor |
|-----------|-------|
| **Protocol** | http |
| **Server Name** | localhost |
| **Port** | 8080 |
| **HTTP Request** | GET |
| **Path** | /api/v1/items |

### Ejemplos de Pruebas Recomendadas

#### Prueba 1: Carga Simple (100 usuarios, 5 minutos)
```
Thread Group:
- Number of Threads: 100
- Ramp-Up Period: 10 segundos
- Loop Count: 5
- Duration: 300 segundos

HTTP Sampler:
- Protocol: http
- Server: localhost
- Port: 8080
- Path: /api/v1/items
```

#### Prueba 2: Múltiples Endpoints (simulación realista)
```
Crea 3 HTTP Samplers:
1. GET /api/v1/health
2. GET /api/v1/items
3. GET /api/v1/items/report (más lento)
```

#### Prueba 3: Operaciones CRUD
```
4 HTTP Samplers secuenciales:
1. GET /api/v1/items (leer)
2. POST /api/v1/items (crear)
3. PUT /api/v1/items/1 (actualizar)
4. DELETE /api/v1/items/1 (eliminar)
```

## Solución de Problemas

### Error: "Conexión rechazada" en JMeter
**Causa**: La aplicación no está corriendo.
```bash
# Verifica que está ejecutándose
curl http://localhost:8080/api/v1/health
```

### Error: Puerto 8080 ya está en uso
**Solución**: Cambia el puerto editando `src/main/resources/application.properties`:
```properties
server.port=9090
```
Luego recompila y ejecuta.

### Compilación lenta
**Tip**: Usa Maven con más memoria disponible:
```bash
export MAVEN_OPTS="-Xmx512m"
mvn clean package
```

## Notas Importantes

- La aplicación usa una **base de datos en memoria** (List), los datos se pierden al reiniciar
- Ideal para **pruebas de rendimiento** y **pruebas de carga**
- Todos los endpoints responden en **< 100ms** (excepto `/report` que tarda ~2s)
- La aplicación es **thread-safe** para múltiples usuarios concurrentes

## Para Profundizar

Consulta la guía principal en `index.html` para:
- Instalación de JMeter
- Crear test plans
- Análisis de resultados
- Mejores prácticas de pruebas de carga
