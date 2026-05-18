#!/bin/bash

echo "================================================================"
echo "  AutoSpark - Pruebas de Rendimiento con JMeter"
echo "================================================================"

cd ~/universidad-proyecto/tests/jmeter

# Crear carpeta para resultados
DATE=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results/$DATE"
mkdir -p $RESULTS_DIR

echo ""
echo "📋 Configuración:"
echo "   Plan de pruebas: plans/AutoSpark_LoadTest.jmx"
echo "   Resultados: $RESULTS_DIR"
echo ""

# Verificar que los servicios están corriendo
echo "🔍 Verificando servicios..."
cd ~/universidad-proyecto/docker

if ! docker ps | grep -q autospark-backend; then
    echo "⚠️ Servicios no disponibles. Iniciándolos..."
    docker-compose up -d mysql backend frontend selenium-hub chrome firefox
    echo "⏳ Esperando 20 segundos..."
    sleep 20
fi

# Verificar health del backend
if curl -s http://localhost:8080/actuator/health | grep -q "UP"; then
    echo "✅ Backend disponible"
else
    echo "⚠️ Backend no responde, continuando de todas formas..."
fi

echo ""
echo "🚀 Ejecutando pruebas de rendimiento..."
echo "   Esto puede tomar varios minutos..."
echo ""

# Ejecutar JMeter con Docker
docker run --rm \
  -v $(pwd)/plans:/plans \
  -v $(pwd)/$RESULTS_DIR:/results \
  --network docker_autospark-network \
  justb4/jmeter:latest \
  -n -t /plans/AutoSpark_LoadTest.jmx \
  -l /results/test-results.jtl \
  -e -o /results/html-report

echo ""
echo "================================================================"
echo "✅ Pruebas de rendimiento completadas"
echo "================================================================"
echo ""
echo "📊 Reportes generados:"
echo "   - HTML: $(pwd)/$RESULTS_DIR/html-report/index.html"
echo "   - JTL:  $(pwd)/$RESULTS_DIR/test-results.jtl"
echo ""
echo "📁 Para copiar a Windows y ver el reporte:"
echo "   cp -r $(pwd)/$RESULTS_DIR/html-report /mnt/c/Users/veraj/Desktop/jmeter-reporte"
echo ""
echo "   Luego abre en Windows: C:\\Users\\veraj\\Desktop\\jmeter-reporte\\index.html"
echo "================================================================"
