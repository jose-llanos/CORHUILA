# Dockerfile multistage para compilar y ejecutar la aplicación Spring Boot (Java 11).

FROM maven:3.9-eclipse-temurin-11 AS build
WORKDIR /workspace

# Copiamos primero el POM para aprovechar la cache de dependencias cuando sea posible.
COPY app/pom.xml app/pom.xml
COPY app/src app/src

RUN mvn -f app/pom.xml -DskipTests package

FROM eclipse-temurin:11-jre AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY --from=build /workspace/app/target/*.jar /app/app.jar

EXPOSE 8080
ENTRYPOINT ["java","-jar","/app/app.jar"]
