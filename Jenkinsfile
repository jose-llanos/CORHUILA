pipeline {
    agent any

    environment {
        PROJECT_DIR = '/workspace'
        HOST_PROJECT_DIR = '/home/juliana/automatizacion-pruebas'
        BACKEND_DIR = '/workspace/app'
        FRONTEND_DIR = '/workspace/frontend'
        JMETER_TEST = '/tests/test-plan.jmx'
        SONAR_HOST_URL = 'http://autospark_sonarqube:9000'
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

        stage('Preparar datos de prueba') {
            steps {
                sh '''
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
                    "
                '''
            }
        }

        stage('Compilar backend') {
            steps {
                dir("${BACKEND_DIR}") {
                    sh 'mvn clean package -DskipTests'
                }
            }
        }

        stage('Pruebas unitarias JUnit + JaCoCo') {
            steps {
                dir("${BACKEND_DIR}") {
                    sh 'mvn test jacoco:report'
                }
            }
        }

        stage('Análisis SonarQube') {
            steps {
                dir("${BACKEND_DIR}") {
                    withCredentials([string(credentialsId: 'sonar-token', variable: 'SONAR_TOKEN')]) {
                        sh '''
                            mvn sonar:sonar \
                              -Dsonar.projectKey=autospark \
                              -Dsonar.projectName=autospark \
                              -Dsonar.host.url=$SONAR_HOST_URL \
                              -Dsonar.token=$SONAR_TOKEN
                        '''
                    }
                }
            }
        }

        stage('Pruebas funcionales Selenium') {
            steps {
                dir("${PROJECT_DIR}/tests/selenium") {
                    sh '''
                        if [ -f pom.xml ]; then
                            mvn test
                        else
                            echo "No se encontró pom.xml en tests/selenium"
                            ls -la
                        fi
                    '''
                }
            }
        }

        stage('Pruebas de rendimiento JMeter') {
            steps {
                sh '''
                    mkdir -p $HOST_PROJECT_DIR/reports/jmeter

                    NETWORK=$(docker inspect autospark_backend --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}')

                    docker run --rm \
                      --network "$NETWORK" \
                      -v $HOST_PROJECT_DIR/tests/jmeter:/tests \
                      -v $HOST_PROJECT_DIR/reports/jmeter:/reports \
                      justb4/jmeter \
                      -n \
                      -t /tests/test-plan.jmx \
                      -l /reports/resultados.jtl \
                      -e \
                      -o /reports/html
                '''
            }
        }

        stage('Verificar reportes') {
            steps {
                sh '''
                    echo "Reportes generados:"
                    ls -la $PROJECT_DIR/reports || true
                    ls -la $PROJECT_DIR/reports/jmeter || true
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
            echo 'El pipeline falló. Revisar la etapa marcada en rojo.'
        }
    }
}