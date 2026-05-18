pipeline {
    agent any

    environment {
        PROJECT_DIR = '/workspace'
        HOST_PROJECT_DIR = '/home/juliana/automatizacion-pruebas'
        BACKEND_DIR = '/workspace/app'
        FRONTEND_DIR = '/workspace/frontend'
        DOCKER_DIR = '/workspace/docker'
        JMETER_TEST = '/tests/AutoSpark_LoadTest.jmx'
        SONAR_HOST_URL = 'http://host.docker.internal:9000'
        MAVEN_OPTS_RETRY = '-Dmaven.wagon.http.retryHandler.count=5 -Dmaven.wagon.httpconnectionManager.ttlSeconds=120'
    }

    stages {

        stage('Verificar entorno') {
            steps {
                sh '''
                    echo "Workspace:"
                    ls -la $PROJECT_DIR

                    echo "Docker:"
                    docker ps

                    echo "Maven:"
                    mvn -version
                '''
            }
        }

        stage('Levantar SonarQube') {
            steps {
                dir("${DOCKER_DIR}") {
                    sh '''
                        echo "Levantando SonarQube..."
                        docker compose up -d sonarqube

                        echo "Esperando SonarQube..."
                        sleep 60

                        docker ps
                        curl -I $SONAR_HOST_URL || true
                    '''
                }
            }
        }

        stage('Compilar backend') {
            steps {
                dir("${BACKEND_DIR}") {
                    sh '''
                        mvn clean package \
                          -DskipTests \
                          $MAVEN_OPTS_RETRY
                    '''
                }
            }
        }

        stage('Pruebas unitarias JUnit + JaCoCo') {
            steps {
                dir("${BACKEND_DIR}") {
                    sh '''
                        mvn test jacoco:report \
                          $MAVEN_OPTS_RETRY
                    '''
                }
            }
        }

        stage('Analisis SonarQube') {
            steps {
                dir("${BACKEND_DIR}") {
                    withCredentials([string(credentialsId: 'sonar-token', variable: 'SONAR_TOKEN')]) {
                        sh '''
                            mvn sonar:sonar \
                              -Dsonar.projectKey=autospark \
                              -Dsonar.projectName=autospark \
                              -Dsonar.host.url=$SONAR_HOST_URL \
                              -Dsonar.token=$SONAR_TOKEN \
                              $MAVEN_OPTS_RETRY
                        '''
                    }
                }
            }
        }

        stage('Levantar aplicacion') {
            steps {
                dir("${DOCKER_DIR}") {
                    sh '''
                        echo "Levantando contenedores..."
                        docker compose up -d --build mysql backend frontend sonarqube

                        echo "Esperando frontend y backend..."
                        sleep 40

                        echo "Contenedores activos:"
                        docker ps

                        echo "Validando frontend dentro de Docker:"
                        curl -I http://autospark_frontend:80 || true

                        echo "Validando backend dentro de Docker:"
                        curl -H "Host: localhost:8080" http://autospark_backend:8080/autospark/service || true
                    '''
                }
            }
        }

        stage('Preparar datos de prueba') {
            steps {
                sh '''
                    echo "Insertando datos de prueba en MySQL..."

                    docker exec autospark_mysql mysql -uroot -proot autospark -e "
                    INSERT INTO usuarios 
                    (fullname, identity_card, email, phone, contrasenia, licenseplate, role)
                    SELECT 
                        'Juan Perez',
                        '12345678',
                        'juan@test.com',
                        '3001234567',
                        SHA2('Password123', 256),
                        'ABC123',
                        'CUSTOMER'
                    WHERE NOT EXISTS (
                        SELECT 1 FROM usuarios WHERE email = 'juan@test.com'
                    );

                    INSERT INTO services (name_service, descripcion, precio_service, estado_service, url_img)
                    SELECT 'Lavado Basico', 'Lavado exterior e interior basico', 25000.00, true, 'https://media.istockphoto.com/id/451803261/es/foto/el-lavado-de.jpg?s=612x612&w=0&k=20&c=eZG1tYFx-xrYi81oAq7mVSps6aKE--A6qWI7f9XMNtU='
                    WHERE NOT EXISTS (SELECT 1 FROM services WHERE name_service = 'Lavado Basico');

                    INSERT INTO services (name_service, descripcion, precio_service, estado_service, url_img)
                    SELECT 'Lavado Premium', 'Lavado completo con cera y aspirado', 45000.00, true, 'https://static.retail.autofact.cl/blog/c_img_740x370.pu9xe7m8lkiqrxxr.jpg'
                    WHERE NOT EXISTS (SELECT 1 FROM services WHERE name_service = 'Lavado Premium');

                    INSERT INTO services (name_service, descripcion, precio_service, estado_service, url_img)
                    SELECT 'Pulido', 'Pulido de carroceria profesional', 80000.00, true, 'https://img.freepik.com/foto-gratis/estacion-lavado-autos_1303-22307.jpg?semt=ais_hybrid&w=740'
                    WHERE NOT EXISTS (SELECT 1 FROM services WHERE name_service = 'Pulido');

                    INSERT INTO services (name_service, descripcion, precio_service, estado_service, url_img)
                    SELECT 'Lavado de Motor', 'Limpieza profunda del motor', 35000.00, true, 'https://media.istockphoto.com/id/1135252747/es/foto/detalles-del-coche-motor-de-limpieza-de-lavado-de-coches-limpieza-del-coche-con-vapor-caliente.jpg?s=612x612&w=0&k=20&c=t4--WADIUPFV6pX9b0KnP7CF3CzlZ_AlNQg28ahMOP0='
                    WHERE NOT EXISTS (SELECT 1 FROM services WHERE name_service = 'Lavado de Motor');

                    INSERT INTO services (name_service, descripcion, precio_service, estado_service, url_img)
                    SELECT 'Lavado de Tapiceria', 'Limpieza profunda de asientos y alfombras', 60000.00, true, 'https://media.istockphoto.com/id/481395168/es/foto/hombre-hoovering-asiento-de-coche-del-autom%C3%B3vil-de-limpieza.jpg?s=612x612&w=0&k=20&c=CtITYKyO4rNvbwPbS1NLIJwLVsCe7ZVjPSmYxUlk8Cs='
                    WHERE NOT EXISTS (SELECT 1 FROM services WHERE name_service = 'Lavado de Tapiceria');
                    "

                    echo "Validando servicios en MySQL:"
                    docker exec autospark_mysql mysql -uroot -proot autospark -e "SELECT id_servicios, name_service FROM services;"

                    echo "Validando servicios desde backend:"
                    curl http://autospark_backend:8080/autospark/service || true
                '''
            }
        }

        stage('Pruebas de rendimiento JMeter') {
            steps {
                sh '''
                    echo "Limpiando reportes anteriores..."

                    rm -rf $HOST_PROJECT_DIR/reports/jmeter
                    mkdir -p $HOST_PROJECT_DIR/reports/jmeter/html

                    mkdir -p $HOST_PROJECT_DIR/reports/jmeter

                    NETWORK=$(docker inspect autospark_backend --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}')

                    echo "Red Docker detectada: $NETWORK"

                    docker run --rm --network $NETWORK \
                      -v $HOST_PROJECT_DIR/tests/jmeter:/tests \
                      -v $HOST_PROJECT_DIR/reports/jmeter:/reports \
                      justb4/jmeter \
                      -n -t $JMETER_TEST \
                      -l /reports/resultados.jtl \
                      -e -o /reports/html
                '''
            }
        }

        stage('Pruebas funcionales Selenium') {
            steps {
                dir("${PROJECT_DIR}/tests/selenium") {
                    sh '''
                        if [ -f pom.xml ]; then
                            mvn test $MAVEN_OPTS_RETRY
                        else
                            echo "No se encontro pom.xml en tests/selenium"
                            ls -la
                        fi
                    '''
                }
            }
        }

        stage('Verificar reportes') {
            steps {
                sh '''
                    echo "Reportes generados:"
                    ls -la $HOST_PROJECT_DIR/reports || true
                    ls -la $HOST_PROJECT_DIR/reports/jmeter || true
                    ls -la $HOST_PROJECT_DIR/reports/jmeter/html || true
                    ls -la $BACKEND_DIR/target/site/jacoco || true
                '''
            }
        }
    }

    post {
        success {
            echo 'Pipeline ejecutado correctamente: JUnit, Selenium, JMeter y SonarQube completados.'
        }

        failure {
            echo 'El pipeline fallo.'
        }
    }
}