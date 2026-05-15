
# =====================================================================
# SGP LAB Backend - Dockerfile multi-stage
# =====================================================================
 
# ---------- STAGE 1: build ----------
FROM maven:3.9-eclipse-temurin-17 AS builder
 
WORKDIR /build
 
# Copiar solo el pom primero para cachear las dependencias
COPY pom.xml ./
RUN mvn -B dependency:go-offline -DskipTests
 
# Copiar el resto del codigo y empaquetar
COPY src ./src
 
# Saltamos tests y javadoc dentro del contenedor (los tests corren en CI).
# Renombramos explicitamente el jar ejecutable (no el .original ni el -javadoc).
RUN mvn -B clean package -DskipTests -Dmaven.javadoc.skip=true \
    && JAR_FILE=$(ls target/*.jar | grep -v "\.original$" | grep -v "javadoc" | head -n 1) \
    && cp "$JAR_FILE" target/app.jar \
    && ls -la target/app.jar
 
# ---------- STAGE 2: runtime ----------
FROM eclipse-temurin:17-jre-jammy AS runtime
 
RUN groupadd --system --gid 1001 sgp \
    && useradd --system --uid 1001 --gid sgp --shell /bin/false --no-create-home sgp
 
WORKDIR /app
 
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
 
COPY --from=builder --chown=sgp:sgp /build/target/app.jar /app/app.jar
 
USER sgp
 
EXPOSE 8080
 
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD curl --fail http://localhost:8080/actuator/health || exit 1
 
ENV JAVA_OPTS=""
 
ENTRYPOINT ["sh", "-c", "exec java $JAVA_OPTS -jar /app/app.jar"]
 