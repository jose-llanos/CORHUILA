# Stage 1: Build
FROM eclipse-temurin:21-jdk-alpine AS build
WORKDIR /build

COPY backend/.mvn .mvn
COPY backend/mvnw .
COPY backend/pom.xml .
RUN chmod +x mvnw && ./mvnw dependency:go-offline -q

COPY backend/src src
RUN ./mvnw clean package -DskipTests

# Stage 2: Runtime
FROM eclipse-temurin:21-jre-alpine
WORKDIR /app
COPY --from=build /build/target/medicita-0.0.1-SNAPSHOT.jar app.jar
EXPOSE 8080
ENV JAVA_OPTS=""
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
