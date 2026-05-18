# 🥖 Sistema Web de Gestión para Panadería Dulce Pan

## 📌 Descripción del Proyecto

Este proyecto corresponde al desarrollo de un sistema web de gestión para una panadería, desarrollado como parte de la asignatura de Calidad de Software.

La aplicación fue construida bajo una arquitectura cliente-servidor, permitiendo gestionar productos, controlar inventario y administrar el carrito de compras. Además, se implementó una estrategia integral de pruebas de software para validar funcionalidad, integración, rendimiento y calidad del código.

---

# 🏗 Arquitectura del Sistema

La solución fue desarrollada bajo una arquitectura cliente-servidor:

## Frontend

Desarrollado con:

- Angular
- TypeScript
- HTML5
- CSS3

Permite:

- Consultar productos
- Visualizar inventario
- Agregar productos al carrito
- Navegar entre módulos

## Backend

Desarrollado con:

- Java 17
- Spring Boot
- Spring Data JPA
- Hibernate
- Maven

Permite:

- Exponer API REST
- Gestionar productos
- Gestionar carrito de compras
- Conectarse con la base de datos

## Base de Datos

- MySQL

---

# 🛠 Tecnologías Utilizadas

| Tecnología | Uso |
|------------|------|
| Java 17 | Desarrollo backend |
| Spring Boot | API REST |
| Angular | Desarrollo frontend |
| MySQL | Persistencia de datos |
| Maven | Gestión de dependencias |
| JUnit 5 | Pruebas unitarias |
| Mockito | Simulación de dependencias |
| H2 Database | Pruebas de integración |
| Selenium WebDriver | Pruebas funcionales |
| Apache JMeter | Pruebas de rendimiento |
| JaCoCo | Cobertura de código |
| SonarQube | Calidad de código |
| Docker | Contenerización |
| GitHub Actions | Integración continua |

---


# ⚙ Funcionalidades Implementadas

## 🛒 Gestión de Productos
- Consultar productos
- Consultar producto por ID
- Crear productos
- Actualizar productos
- Eliminar productos

## 🛍 Gestión del Carrito
- Agregar productos
- Consultar carrito
- Actualización de cantidades
- Eliminar productos del carrito
- Vaciar carrito

---

# 🧪 Estrategia de Pruebas

## 1. Pruebas Unitarias

### Herramientas utilizadas
- JUnit 5
- Mockito
- JaCoCo

### Cobertura obtenida
- 91% de cobertura

### Componentes probados
- Servicios de productos
- Servicios de carrito
- Controladores REST

---

## 2. Pruebas de Integración

### Herramientas utilizadas
- Spring Boot Test
- H2 Database

### Validaciones realizadas
- Integración entre servicios
- Persistencia en base de datos
- Operaciones CRUD

---

## 3. Pruebas Funcionales

### Herramientas utilizadas
- Selenium WebDriver
- Page Object Model (POM)

### Escenarios evaluados
- Apertura del módulo productos
- Apertura del carrito
- Apertura del inventario
- Navegación entre módulos

### Resultado obtenido
- 5 pruebas exitosas

---

## 4. Pruebas de Rendimiento

### Herramienta utilizada
- Apache JMeter

### Escenarios de prueba
- 10 usuarios concurrentes
- Pruebas sobre endpoints REST


---
### Endpoints evaluados
#### Productos
```http
GET     /api/producto
POST    /api/producto
PUT     /api/producto/{id}
DELETE  /api/producto/{id}
```

---

### Métricas analizadas
- Tiempo de respuesta
- Throughput
- Tasa de error

---

## 5. Análisis de Calidad

## Herramienta utilizada
- SonarQube

## Métricas analizadas
- Cobertura
- Code Smells
- Vulnerabilidades
- Duplicación de código
- Security Rating

## Resultado obtenido
- ✅ Quality Gate: Aprobado

---

# 🚀 Ejecución del Proyecto

## Backend

### Ejecutar servidor Spring Boot
```bash
mvn spring-boot:run
```

Disponible en:

```text
http://localhost:8080
```

---

## Frontend

### Ejecutar aplicación Angular
```bash
ng serve
```

Disponible en:

```text
http://localhost:4200
```

---

# ▶ Ejecutar Pruebas

## Pruebas unitarias + integración
```bash
mvn clean test
```

## Generar cobertura con JaCoCo
```bash
mvn jacoco:report
```

## Ejecutar pruebas Selenium

> Importante: el backend y frontend deben estar ejecutándose.

```bash
mvn test -Dtest=PanaderiaSeleniumTest
```

---

# 👥 Autores

Proyecto académico — Calidad de Software

- Seam Yong
- Alex Salgado

---

# 🎓 Institución

Corporación Universitaria del Huila — CORHUILA

---

# 📌 Estado del Proyecto

- ✅ Proyecto finalizado
- ✅ Validado mediante pruebas de software
- ✅ Documentado y versionado en GitHub

---

# 📂 Estructura del Repositorio

```bash
.
├── app/
│   ├── src/main
│   └── src/test
│
├── tests/
│   ├── selenium/
│   └── jmeter/
│
├── reports/
│
├── docs/
│
├── pom.xml
└── README.md
