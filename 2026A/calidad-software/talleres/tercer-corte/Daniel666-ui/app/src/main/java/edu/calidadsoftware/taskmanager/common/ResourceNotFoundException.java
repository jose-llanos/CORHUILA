package edu.calidadsoftware.taskmanager.common;

/**
 * Excepción de dominio para recursos inexistentes (por ejemplo, Task o User con id inválido).
 *
 * Se maneja en un @ControllerAdvice para devolver páginas de error (MVC) o respuestas JSON (API).
 */
public class ResourceNotFoundException extends RuntimeException {

    public ResourceNotFoundException(String message) {
        super(message);
    }
}
