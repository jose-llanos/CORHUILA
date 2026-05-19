package edu.calidadsoftware.taskmanager;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Punto de entrada de la aplicación Spring Boot.
 *
 * Esta clase arranca el contenedor embebido (Tomcat) y habilita el escaneo de componentes
 * para controladores, servicios, repositorios y configuración.
 */
@SpringBootApplication
public class TaskManagerApplication {

    public static void main(String[] args) {
        SpringApplication.run(TaskManagerApplication.class, args);
    }
}
