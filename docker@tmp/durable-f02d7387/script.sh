
                        echo "Levantando SonarQube..."
                        docker compose up -d sonarqube

                        echo "Esperando SonarQube..."
                        sleep 60

                        docker ps
                        curl -I $SONAR_HOST_URL || true
                    