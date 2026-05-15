package com.sgplab.backend.exception;

/**
 * Excepcion lanzada cuando una operacion viola una regla de negocio del dominio
 * (por ejemplo, intentar crear un prestamo con stock insuficiente).
 * Se traduce automaticamente a HTTP 409 Conflict.
 *
 * @author SGP LAB Team
 */
public class BusinessRuleException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public BusinessRuleException(String message) {
        super(message);
    }
}
