#!/usr/bin/env sh

set -eu

# Ejecuta el análisis SonarQube usando Maven dentro de un contenedor.
# Requisitos:
# - docker compose up (sonarqube levantado en la red taskmanager-network)
# - En SonarQube, para experiencia "out-of-the-box" en local, se configura sonar.forceAuthentication=false.
#
# Uso:
#   ./docker/sonar-scan.sh
#   SONAR_TOKEN=xxxxx ./docker/sonar-scan.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SONAR_HOST_URL="${SONAR_HOST_URL:-http://sonarqube:9000}"
SONAR_PROJECT_KEY="${SONAR_PROJECT_KEY:-taskmanager}"

SONAR_LOGIN_ARGS=""
if [ "${SONAR_TOKEN:-}" != "" ]; then
  SONAR_LOGIN_ARGS="-Dsonar.login=${SONAR_TOKEN}"
fi

docker run --rm \
  --network taskmanager-network \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace/app \
  maven:3.9-eclipse-temurin-11 \
  mvn verify sonar:sonar \
    -Dsonar.host.url="${SONAR_HOST_URL}" \
    -Dsonar.projectKey="${SONAR_PROJECT_KEY}" \
    ${SONAR_LOGIN_ARGS}
