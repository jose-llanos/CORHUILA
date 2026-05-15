package com.sgplab.backend.exception;

/**
 * Excepcion lanzada cuando se intenta crear un recurso que ya existe
 * (por ejemplo, dos usuarios con el mismo email).
 * Se traduce a HTTP 409 Conflict.
 *
 * @author SGP LAB Team
 */
public class DuplicateResourceException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public DuplicateResourceException(String message) {
        super(message);
    }
}
