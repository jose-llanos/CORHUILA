#!/usr/bin/env sh

set -eu

# Script de inicialización:
# - Espera a que los servicios estén listos mediante healthchecks.
# - Ejecuta el análisis SonarQube automáticamente.
# - Imprime URLs de acceso al finalizar.

COMPOSE_FILE="docker-compose.yml"

wait_healthy() {
  service_name="$1"
  echo "Esperando healthcheck de: ${service_name}"
  i=1
  while [ "$i" -le 120 ]; do
    cid="$(docker compose -f "${COMPOSE_FILE}" ps -q "${service_name}" 2>/dev/null || true)"
    if [ "${cid}" != "" ]; then
      status="$(docker inspect -f "{{.State.Health.Status}}" "${cid}" 2>/dev/null || true)"
      if [ "${status}" = "healthy" ]; then
        echo "OK: ${service_name} está healthy"
        return 0
      fi
    fi
    sleep 3
    i=$((i + 1))
  done
  echo "ERROR: timeout esperando healthcheck de ${service_name}"
  return 1
}

wait_healthy_optional() {
  service_name="$1"
  cid="$(docker compose -f "${COMPOSE_FILE}" ps -q "${service_name}" 2>/dev/null || true)"
  if [ "${cid}" = "" ]; then
    echo "SKIP: ${service_name} no está iniciado"
    return 0
  fi
  wait_healthy "${service_name}"
}

wait_healthy "sonarqube"
wait_healthy "app"
wait_healthy_optional "selenium-hub"
wait_healthy_optional "chrome"
wait_healthy_optional "firefox"

echo "Ejecutando análisis SonarQube..."
sh docker/sonar-scan.sh || true

echo ""
echo "Servicios disponibles:"
echo "- App:        http://localhost:8080"
echo "- SonarQube:   http://localhost:9000"
echo "- Selenium UI: http://localhost:4444"
