#!/bin/bash

echo "================================================================"
echo "  AutoSpark - Pruebas de Rendimiento con JMeter"
echo "================================================================"

cd ~/universidad-proyecto/tests/jmeter

# 1. LIMPIAR RESULTADOS ANTERIORES
echo ""
echo "🧹 Limpiando resultados anteriores..."
sudo rm -rf results/* 2>/dev/null
mkdir -p results

# 2. INSERTAR USUARIO DE PRUEBA
echo ""
echo "📝 Insertando usuario de prueba en la base de datos..."
echo "   Email: juan@test.com"
echo "   Password: Password123"

# Eliminar usuarios existentes
docker exec autospark-mysql mysql -uautospark_user -pautospark_password autospark -e "
DELETE FROM usuarios WHERE email = 'juan@test.com';
" 2>/dev/null

# Insertar usuario nuevo
docker exec autospark-mysql mysql -uautospark_user -pautospark_password autospark -e "
INSERT INTO usuarios (fullname, identity_card, email, phone, contrasenia, licenseplate, role) 
VALUES ('Juan Perez', '12345678', 'juan@test.com', '3001234567', SHA2('Password123', 256), 'ABC123', 'CUSTOMER');
" 2>/dev/null

echo "✅ Usuario insertado correctamente"

# 3. VERIFICAR QUE EL BACKEND RESPONDE
echo ""
echo "🔍 Verificando backend..."
if curl -s http://localhost:8080/actuator/health | grep -q "UP"; then
    echo "✅ Backend disponible"
else
    echo "❌ Backend no disponible"
    exit 1
fi

# 4. CREAR PLAN JMETER CON EL DOMINIO CORRECTO
echo ""
echo "📝 Creando plan JMeter con dominio 'backend'..."

cat > plans/corrected-load-test.jmx << 'JMXEOF'
<?xml version="1.0" encoding="UTF-8"?>
<jmeterTestPlan version="1.2" properties="5.0" jmeter="5.6.3">
  <hashTree>
    <TestPlan guiclass="TestPlanGui" testclass="TestPlan" testname="AutoSpark Performance Test" enabled="true">
      <boolProp name="TestPlan.functional_mode">false</boolProp>
      <boolProp name="TestPlan.tearDown_on_shutdown">true</boolProp>
    </TestPlan>
    <hashTree>
      <ConfigTestElement guiclass="HttpDefaultsGui" testclass="ConfigTestElement" testname="HTTP Request Defaults" enabled="true">
        <stringProp name="HTTPSampler.domain">backend</stringProp>
        <stringProp name="HTTPSampler.port">8080</stringProp>
        <stringProp name="HTTPSampler.protocol">http</stringProp>
      </ConfigTestElement>
      <hashTree/>
      <HeaderManager guiclass="HeaderPanel" testclass="HeaderManager" testname="HTTP Header Manager" enabled="true">
        <collectionProp name="HeaderManager.headers">
          <elementProp name="" elementType="Header">
            <stringProp name="Header.name">Content-Type</stringProp>
            <stringProp name="Header.value">application/json</stringProp>
          </elementProp>
        </collectionProp>
      </HeaderManager>
      <hashTree/>
      <ThreadGroup guiclass="ThreadGroupGui" testclass="ThreadGroup" testname="Normal - 5 usuarios" enabled="true">
        <stringProp name="ThreadGroup.num_threads">5</stringProp>
        <stringProp name="ThreadGroup.ramp_time">5</stringProp>
        <stringProp name="ThreadGroup.loops">10</stringProp>
      </ThreadGroup>
      <hashTree>
        <HTTPSamplerProxy guiclass="HttpTestSampleGui" testclass="HTTPSamplerProxy" testname="POST Login" enabled="true">
          <elementProp name="HTTPsampler.Arguments" elementType="Arguments">
            <collectionProp name="Arguments.arguments">
              <elementProp name="" elementType="HTTPArgument">
                <stringProp name="HTTPArgument.value">{"email":"juan@test.com","password":"Password123"}</stringProp>
              </elementProp>
            </collectionProp>
          </elementProp>
          <stringProp name="HTTPSampler.path">/autospark/login</stringProp>
          <stringProp name="HTTPSampler.method">POST</stringProp>
        </HTTPSamplerProxy>
        <hashTree/>
        <HTTPSamplerProxy guiclass="HttpTestSampleGui" testclass="HTTPSamplerProxy" testname="GET Servicios" enabled="true">
          <stringProp name="HTTPSampler.path">/autospark/service</stringProp>
          <stringProp name="HTTPSampler.method">GET</stringProp>
        </HTTPSamplerProxy>
        <hashTree/>
        <HTTPSamplerProxy guiclass="HttpTestSampleGui" testclass="HTTPSamplerProxy" testname="GET Usuarios" enabled="true">
          <stringProp name="HTTPSampler.path">/autospark/users</stringProp>
          <stringProp name="HTTPSampler.method">GET</stringProp>
        </HTTPSamplerProxy>
        <hashTree/>
        <HTTPSamplerProxy guiclass="HttpTestSampleGui" testclass="HTTPSamplerProxy" testname="GET Reservas" enabled="true">
          <stringProp name="HTTPSampler.path">/autospark/reserva</stringProp>
          <stringProp name="HTTPSampler.method">GET</stringProp>
        </HTTPSamplerProxy>
        <hashTree/>
      </hashTree>
    </hashTree>
  </hashTree>
</jmeterTestPlan>
JMXEOF

echo "✅ Plan JMeter creado"

# 5. EJECUTAR JMETER
echo ""
echo "🚀 Ejecutando pruebas de rendimiento con JMeter..."
echo ""

docker run --rm \
  -v $(pwd)/plans:/plans \
  -v $(pwd)/results:/results \
  --network docker_autospark-network \
  justb4/jmeter:latest \
  -n -t /plans/corrected-load-test.jmx \
  -l /results/test-results.jtl \
  -e -o /results/html-report

# 6. CAMBIAR PROPIETARIO DE RESULTADOS
sudo chown -R jenkins_user:jenkins_user results/ 2>/dev/null

# 7. MOSTRAR RESULTADOS
echo ""
echo "================================================================"
echo "📊 RESULTADOS DE LAS PRUEBAS"
echo "================================================================"

if [ -f results/test-results.jtl ]; then
    TOTAL=$(cat results/test-results.jtl | wc -l)
    SUCCESS=$(grep -c ",true," results/test-results.jtl 2>/dev/null)
    FAILED=$(grep -c ",false," results/test-results.jtl 2>/dev/null)
    
    echo ""
    echo "📈 MÉTRICAS GENERALES:"
    echo "   Total peticiones: $TOTAL"
    echo "   Peticiones exitosas: $SUCCESS"
    echo "   Peticiones fallidas: $FAILED"
    
    if [ $TOTAL -gt 0 ]; then
        SUCCESS_RATE=$(echo "scale=2; $SUCCESS * 100 / $TOTAL" | bc)
        echo "   Tasa de éxito: ${SUCCESS_RATE}%"
    fi
else
    echo "❌ No se encontró el archivo de resultados"
fi

# 8. COPIAR REPORTE A WINDOWS
echo ""
echo "📁 Copiando reporte a Windows..."
cp -r results/html-report /mnt/c/Users/veraj/Desktop/jmeter-reporte-final 2>/dev/null

echo ""
echo "================================================================"
echo "✅ PRUEBAS COMPLETADAS"
echo "================================================================"
echo "📊 Reporte HTML: C:\\Users\\veraj\\Desktop\\jmeter-reporte-final\\index.html"
echo "================================================================"
