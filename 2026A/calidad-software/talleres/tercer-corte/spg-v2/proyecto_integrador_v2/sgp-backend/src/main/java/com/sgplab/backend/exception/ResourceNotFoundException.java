package com.sgplab.backend.exception;

/**
 * Excepcion lanzada cuando un recurso solicitado no existe en el sistema.
 * Se traduce automaticamente a HTTP 404 por el manejador global.
 *
 * @author SGP LAB Team
 */
public class ResourceNotFoundException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public ResourceNotFoundException(String message) {
        super(message);
    }

    public ResourceNotFoundException(String resourceName, String fieldName, Object fieldValue) {
        super(String.format("%s no encontrado con %s: '%s'", resourceName, fieldName, fieldValue));
    }
}
