package com.sgplab.backend.exception;

/**
 * Excepcion lanzada cuando un intento de autenticacion falla
 * (email no existe o contrasena incorrecta).
 * Se traduce automaticamente a HTTP 401 Unauthorized.
 *
 * @author SGP LAB Team
 */
public class InvalidCredentialsException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public InvalidCredentialsException(String message) {
        super(message);
    }
}
