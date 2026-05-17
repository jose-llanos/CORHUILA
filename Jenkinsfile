pipeline {
    agent any

    stages {

        stage('Verificar workspace') {
            steps {
                sh 'ls -la /workspace'
            }
        }

        stage('Verificar Docker') {
            steps {
                sh 'docker ps'
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
                        SELECT 1 
                        FROM usuarios 
                        WHERE email = 'juan@test.com'
                    );
                    "
                '''
            }
        }

        stage('Compilar backend') {
            steps {
                dir('/workspace/app') {
                    sh 'mvn clean package -DskipTests'
                }
            }
        }

        stage('Pruebas unitarias JUnit') {
            steps {
                dir('/workspace/app') {
                    sh 'mvn test'
                }
            }
        }

        stage('Verificar contenedores') {
            steps {
                sh 'docker ps'
            }
        }
    }

    post {
        success {
            echo 'Pipeline ejecutado correctamente'
        }

        failure {
            echo 'El pipeline falló'
        }
    }
}