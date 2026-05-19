# Imagen de Jenkins con JDK 21 + Maven, necesaria para correr el pipeline E2E.
#
# Motivo: jenkins/jenkins:lts trae JDK 17 por defecto, pero el proyecto compila
# con Java 21 (spring-boot-starter-parent 4.0.6 lo exige). Sin JDK 21 dentro
# del contenedor de Jenkins, ./mvnw test -Pe2e fallaría.
#
# Mantenemos el resto del setup oficial intacto.

FROM jenkins/jenkins:lts

USER root

# Docker CLI (no daemon, solo cliente) para usar el socket montado
RUN install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/debian/gpg \
        -o /etc/apt/keyrings/docker.asc \
 && chmod a+r /etc/apt/keyrings/docker.asc \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
        > /etc/apt/sources.list.d/docker.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce-cli \
 && rm -rf /var/lib/apt/lists/*

# Añadir usuario jenkins al grupo docker (GID del host)
ARG DOCKER_GID=1000
RUN if getent group ${DOCKER_GID} >/dev/null; then \
        EXISTING_GROUP=$(getent group ${DOCKER_GID} | cut -d: -f1); \
        usermod -aG ${EXISTING_GROUP} jenkins; \
    else \
        groupadd -g ${DOCKER_GID} docker && usermod -aG docker jenkins; \
    fi

# JAVA_HOME apunta al JDK 21 recién instalado.
ENV JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Vuelve al usuario jenkins para correr el proceso (buenas prácticas).
USER jenkins
