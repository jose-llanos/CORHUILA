SHELL := /bin/sh
COMPOSE := docker compose -f docker-compose.yml

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d --build
	sh docker/init.sh

down:
	$(COMPOSE) down -v

logs:
	$(COMPOSE) logs -f --no-color app

test:
	docker run --rm --network taskmanager-network -v "$$(pwd):/workspace" -w /workspace/app maven:3.9-eclipse-temurin-11 mvn test

selenium:
	docker run --rm --network taskmanager-network -v "$$(pwd):/workspace" -w /workspace maven:3.9-eclipse-temurin-11 mvn -f tests/selenium/pom.xml test -DremoteUrl=http://selenium-hub:4444/wd/hub -Dbrowser=chrome -Dheadless=true -DbaseUrl=http://app:8080

sonar:
	sh docker/sonar-scan.sh
