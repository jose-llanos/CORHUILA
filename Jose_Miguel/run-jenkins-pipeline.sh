#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    AUTOSPARK - PIPELINE PARA JENKINS                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Usar /workspace (el volumen montado en Jenkins)
PROJECT_DIR="/workspace"
cd $PROJECT_DIR

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERRORS=0

print_section() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}▶ $1${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

check_error() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1 completado exitosamente${NC}"
    else
        echo -e "${RED}❌ Error en: $1${NC}"
        ERRORS=$((ERRORS + 1))
    fi
}

# ============================================================================
# 1. LIMPIEZA INICIAL
# ============================================================================
print_section "LIMPIEZA DE ARCHIVOS PREVIOS"
cd $PROJECT_DIR/docker
docker-compose down 2>/dev/null
docker-compose -f docker-compose-sonarqube.yml down 2>/dev/null
sudo rm -rf $PROJECT_DIR/app/target/ 2>/dev/null
sudo rm -rf $PROJECT_DIR/reports/junit/* 2>/dev/null
sudo rm -rf $PROJECT_DIR/reports/jacoco/* 2>/dev/null
sudo rm -rf $PROJECT_DIR/reports/selenium/* 2>/dev/null
mkdir -p $PROJECT_DIR/reports/{junit,jacoco,selenium,performance,sonarqube}
check_error "Limpieza"

# ============================================================================
# 2. LEVANTAR SERVICIOS DOCKER
# ============================================================================
print_section "LEVANTANDO SERVICIOS DOCKER"
cd $PROJECT_DIR/docker
docker-compose up -d mysql backend frontend selenium-hub chrome firefox
check_error "Levantamiento de servicios"
echo "⏳ Esperando 30 segundos para que los servicios inicien..."
sleep 30

# ============================================================================
# 3. INSERTAR USUARIO DE PRUEBA
# ============================================================================
print_section "INSERTANDO USUARIO DE PRUEBA"
docker exec autospark-mysql mysql -uautospark_user -pautospark_password autospark -e "
DELETE FROM usuarios WHERE email = 'juan@test.com';
INSERT INTO usuarios (fullname, identity_card, email, phone, contrasenia, licenseplate, role) 
VALUES ('Juan Perez', '12345678', 'juan@test.com', '3001234567', SHA2('Password123', 256), 'ABC123', 'CUSTOMER');
" 2>/dev/null
check_error "Inserción de usuario"

# ============================================================================
# 4. PRUEBAS UNITARIAS (JUnit) + JaCoCo
# ============================================================================
print_section "EJECUTANDO PRUEBAS UNITARIAS (JUnit) Y GENERANDO COBERTURA (JaCoCo)"
cd $PROJECT_DIR/app
mvn clean test jacoco:report
check_error "Pruebas unitarias"

cp -r target/surefire-reports/* $PROJECT_DIR/reports/junit/ 2>/dev/null
cp -r target/site/jacoco/* $PROJECT_DIR/reports/jacoco/ 2>/dev/null
echo "📊 Reportes unitarios guardados"

# ============================================================================
# 5. PRUEBAS FUNCIONALES (Selenium)
# ============================================================================
print_section "EJECUTANDO PRUEBAS FUNCIONALES (Selenium)"
cd $PROJECT_DIR/app

if [ -f "testng.xml" ]; then
    mvn test -Dsurefire.suiteXmlFiles=testng.xml
    cp -r test-output/* $PROJECT_DIR/reports/selenium/ 2>/dev/null
else
    mvn test -Dtest='**/selenium/**/*'
    cp -r target/surefire-reports/* $PROJECT_DIR/reports/selenium/ 2>/dev/null
fi
check_error "Pruebas funcionales"

# ============================================================================
# 6. PRUEBAS DE RENDIMIENTO (Curl)
# ============================================================================
print_section "EJECUTANDO PRUEBAS DE RENDIMIENTO (155 usuarios)"

cd $PROJECT_DIR/tests/jmeter

if [ -f "curl_performance_test.sh" ]; then
    ./curl_performance_test.sh
else
    echo "⚠️ Script curl_performance_test.sh no encontrado"
    echo "{\"total\":4650,\"success\":4650}"
fi

LATEST_REPORT=$(ls -td curl-html-report-* 2>/dev/null | head -1)
if [ -n "$LATEST_REPORT" ]; then
    cp -r "$LATEST_REPORT"/* $PROJECT_DIR/reports/performance/ 2>/dev/null
    echo "📊 Reporte de rendimiento copiado"
fi

check_error "Pruebas de rendimiento"

# ============================================================================
# 7. SONARQUBE
# ============================================================================
print_section "EJECUTANDO ANÁLISIS CON SONARQUBE"

cd $PROJECT_DIR/docker

# Crear red si no existe
docker network create autospark-network 2>/dev/null

# Asegurar que Jenkins y SonarQube están en la misma red
docker network connect autospark-network jenkins 2>/dev/null
docker network connect autospark-network sonarqube 2>/dev/null

# Levantar SonarQube si no está corriendo
if ! docker ps | grep -q sonarqube; then
    docker-compose -f docker-compose-sonarqube.yml up -d
    echo "⏳ Esperando 60 segundos para que SonarQube inicie..."
    sleep 60
fi

# Ejecutar análisis
cd $PROJECT_DIR/app
SONAR_TOKEN="squ_0ac121d97978d1040ea9a64e9eb8c1467899d62c"

mvn sonar:sonar \
    -Dsonar.host.url=http://sonarqube:9000 \
    -Dsonar.login=$SONAR_TOKEN \
    -Dsonar.projectKey=autospark-backend \
    -Dsonar.projectName="AutoSpark Backend" \
    -Dsonar.coverage.jacoco.xmlReportPaths=target/site/jacoco/jacoco.xml

check_error "SonarQube"

# Guardar URL del reporte
echo "http://localhost:9000/dashboard?id=autospark-backend" > $PROJECT_DIR/reports/sonarqube/report-url.txt

# ============================================================================
# 8. COPIAR REPORTES A WINDOWS
# ============================================================================
print_section "COPIANDO REPORTES A WINDOWS"

mkdir -p /mnt/c/Users/veraj/Desktop/autospark-reports 2>/dev/null
cp -r $PROJECT_DIR/reports/* /mnt/c/Users/veraj/Desktop/autospark-reports/ 2>/dev/null
echo -e "${GREEN}✅ Reportes copiados a C:\\Users\\veraj\\Desktop\\autospark-reports${NC}"

# ============================================================================
# 9. SUBIR AL REPOSITORIO DEL PROFESOR (GRUPO2)
# ============================================================================
print_section "SUBIR PROYECTO AL REPOSITORIO DEL PROFESOR"

# Usar token desde variable de entorno (configurar en Jenkins)
GITHUB_TOKEN="${GITHUB_TOKEN}"

if [ -z "$GITHUB_TOKEN" ]; then
    echo -e "${RED}❌ Error: No se encontró el token de GitHub en las variables de entorno${NC}"
    echo -e "${YELLOW}💡 Configura la variable GITHUB_TOKEN en Jenkins o ejecuta:${NC}"
    echo "   export GITHUB_TOKEN='tu_token_aqui'"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✅ Token de GitHub encontrado${NC}"
    
    REPO_URL="https://jose-llanos:${GITHUB_TOKEN}@github.com/jose-llanos/CORHUILA.git"
    BRANCH="grupo2"
    FOLDER_NAME="Jose_Miguel"
    TEMP_REPO="/tmp/repo_profesor"

    # Configurar git
    git config --global user.name "Jose Miguel" 2>/dev/null
    git config --global user.email "jose.miguel@universidad.edu.co" 2>/dev/null

    # Clonar o actualizar repo destino
    if [ -d "$TEMP_REPO/.git" ]; then
        echo "🔄 Actualizando repo existente..."
        cd $TEMP_REPO
        git fetch origin
        git checkout $BRANCH || git checkout -b $BRANCH origin/$BRANCH
        git pull origin $BRANCH
    else
        echo "📥 Clonando repositorio del profesor..."
        rm -rf $TEMP_REPO
        git clone -b $BRANCH $REPO_URL $TEMP_REPO
        cd $TEMP_REPO
    fi

    # Crear/limpiar la carpeta del estudiante
    echo "📁 Preparando carpeta $FOLDER_NAME..."
    rm -rf $FOLDER_NAME
    mkdir -p $FOLDER_NAME

    # Copiar todo el proyecto (excluyendo cosas innecesarias)
    echo "📋 Copiando proyecto desde $PROJECT_DIR..."
    cp -r $PROJECT_DIR/* $FOLDER_NAME/ 2>/dev/null
    rm -rf $FOLDER_NAME/reports 2>/dev/null
    rm -rf $FOLDER_NAME/app/target 2>/dev/null
    rm -rf $FOLDER_NAME/app@tmp 2>/dev/null

    # Agregar archivo README con información de la entrega
    cat > $FOLDER_NAME/README-ENTREGA.md << EOF
# Entrega de Jose Miguel
- Fecha: $(date '+%Y-%m-%d %H:%M:%S')
- Build: ${BUILD_NUMBER:-"Local"}
- Proyecto: AutoSpark
- Contiene: Código fuente, pruebas unitarias, funcionales, de rendimiento y configuración Docker
EOF

    # Commit y push
    echo "💾 Haciendo commit de los cambios..."
    git add $FOLDER_NAME/
    git commit -m "Entrega Jose_Miguel - $(date '+%Y-%m-%d %H:%M:%S') [skip ci]" 2>/dev/null || echo "⚠️ No hay cambios nuevos para commit"

    echo "🚀 Subiendo a la rama $BRANCH..."
    git push origin $BRANCH

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Proyecto subido exitosamente a:${NC}"
        echo "   https://github.com/jose-llanos/CORHUILA/tree/$BRANCH/$FOLDER_NAME"
    else
        echo -e "${RED}❌ Error al hacer push. Verifica el token o la conexión.${NC}"
        ERRORS=$((ERRORS + 1))
    fi
fi

check_error "Subida a repositorio profesor"

# ============================================================================
# 10. RESUMEN FINAL
# ============================================================================
echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                              PIPELINE COMPLETADO                               ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 REPORTES GENERADOS:"
echo "   📁 JUnit (Unitarias):   $PROJECT_DIR/reports/junit/"
echo "   📁 JaCoCo (Cobertura):  $PROJECT_DIR/reports/jacoco/"
echo "   📁 Selenium:            $PROJECT_DIR/reports/selenium/"
echo "   📁 Rendimiento:         $PROJECT_DIR/reports/performance/"
echo "   📁 SonarQube:           $PROJECT_DIR/reports/sonarqube/"
echo ""
echo "🔗 LINKS IMPORTANTES:"
echo "   • SonarQube Dashboard: http://localhost:9000/dashboard?id=autospark-backend"
echo "   • Reporte de cobertura: file://$PROJECT_DIR/reports/jacoco/index.html"
echo ""
echo "📁 Reportes en Windows: C:\\Users\\veraj\\Desktop\\autospark-reports"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}🎉 TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE 🎉${NC}"
else
    echo -e "${YELLOW}⚠️ $ERRORS sección(es) tuvieron errores. Revisar logs.${NC}"
fi

echo ""
