pipeline {
    agent any

    stages {
        stage('Verificar workspace') {
            steps {
                sh 'ls -la /workspace'
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
}
