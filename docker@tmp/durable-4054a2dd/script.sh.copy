
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
                    