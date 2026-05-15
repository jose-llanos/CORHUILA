#!/usr/bin/env bash
# ============================================================
# Ejecuta los planes JMeter en modo no-GUI y genera
# reportes HTML profesionales.
#
# Uso:
#   ./run-jmeter.sh carga      # solo carga normal
#   ./run-jmeter.sh estres     # solo estres
#   ./run-jmeter.sh todo       # ambos (default)
#
# Requiere:
#   - Backend corriendo en http://localhost:8080
#   - JMeter instalado (jmeter en el PATH) o variable JMETER_HOME
# ============================================================

set -euo pipefail

SCENARIO="${1:-todo}"
HOST="${SGP_HOST:-localhost}"
PORT="${SGP_PORT:-8080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}"
RESULTS_DIR="${BASE_DIR}/results"
REPORTS_DIR="${BASE_DIR}/reports"

if command -v jmeter >/dev/null 2>&1; then
  JMETER_CMD="jmeter"
elif [[ -n "${JMETER_HOME:-}" ]] && [[ -x "${JMETER_HOME}/bin/jmeter" ]]; then
  JMETER_CMD="${JMETER_HOME}/bin/jmeter"
else
  echo "ERROR: jmeter no encontrado. Instalalo o exporta JMETER_HOME." >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}" "${REPORTS_DIR}"

run_plan() {
  local plan_name="$1"
  local jmx_file="${BASE_DIR}/${plan_name}.jmx"
  local jtl_file="${RESULTS_DIR}/${plan_name}.jtl"
  local report_dir="${REPORTS_DIR}/${plan_name}"

  echo ""
  echo "=========================================="
  echo " Ejecutando plan: ${plan_name}"
  echo " Host: ${HOST}:${PORT}"
  echo "=========================================="

  rm -f "${jtl_file}"
  rm -rf "${report_dir}"

  "${JMETER_CMD}" \
    -n \
    -t "${jmx_file}" \
    -l "${jtl_file}" \
    -e -o "${report_dir}" \
    -Jhost="${HOST}" \
    -Jport="${PORT}" \
    -Jjmeter.save.saveservice.output_format=csv

  echo ""
  echo "Reporte HTML generado en: ${report_dir}/index.html"
}

case "${SCENARIO}" in
  carga)
    run_plan "carga-normal"
    ;;
  estres)
    run_plan "estres"
    ;;
  todo|all)
    run_plan "carga-normal"
    run_plan "estres"
    ;;
  *)
    echo "Uso: $0 [carga|estres|todo]" >&2
    exit 1
    ;;
esac

echo ""
echo "Todos los reportes en: ${REPORTS_DIR}/"
