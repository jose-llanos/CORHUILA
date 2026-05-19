FROM eclipse-temurin:21-jdk
WORKDIR /app
EXPOSE 8080
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
CMD ["./mvnw", "spring-boot:run", "-Dspring-boot.run.addResources=true","-Dspring-boot.run.fork=false"]

 