package com.calidad.jmeter;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Aplicación de ejemplo para pruebas de carga con JMeter
 * 
 * Esta es una API REST simple con varios endpoints para simular
 * diferentes escenarios de prueba de rendimiento.
 * 
 * Para ejecutar:
 *   1. mvn clean package
 *   2. java -jar target/jmeter-app-ejemplo-1.0.0.jar
 * 
 * La aplicación estará disponible en: http://localhost:8080
 */
@SpringBootApplication
public class ApiApplication {

    public static void main(String[] args) {
        SpringApplication.run(ApiApplication.class, args);
    }
}
