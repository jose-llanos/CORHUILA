# Dockerfile para AutoSpark Backend (Spring Boot)
FROM maven:3.9-eclipse-temurin-17 AS build
WORKDIR /app

# Copiar pom.xml y descargar dependencias
COPY app/pom.xml .
RUN mvn dependency:go-offline

# Copiar código fuente
COPY app/src ./src

# Compilar y generar el JAR (excluyendo pruebas por ahora)
RUN mvn clean package -DskipTests

# Imagen final para ejecutar la aplicación
FROM eclipse-temurin:17-jre-alpine
WORKDIR /app

# Crear usuario no-root para seguridad
RUN addgroup -S spring && adduser -S spring -G spring
USER spring:spring

# Copiar el JAR desde la fase de construcción
COPY --from=build /app/target/migueljuliana-0.0.1-SNAPSHOT.jar app.jar

# Exponer el puerto de la aplicación
EXPOSE 8080

# Comando para ejecutar la aplicación
ENTRYPOINT ["java", "-jar", "app.jar"]

