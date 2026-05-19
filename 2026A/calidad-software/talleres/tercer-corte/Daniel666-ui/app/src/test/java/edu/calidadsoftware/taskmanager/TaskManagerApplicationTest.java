package edu.calidadsoftware.taskmanager;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Prueba simple para ejecutar el método main y aumentar cobertura del punto de entrada.
 */
@DisplayName("TaskManagerApplication")
class TaskManagerApplicationTest {

    @Test
    @DisplayName("main ejecuta sin lanzar excepción")
    void main_runs() {
        TaskManagerApplication.main(new String[]{});
    }
}

