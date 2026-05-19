package edu.calidadsoftware.taskmanager.common;

/**
 * Excepción para indicar duplicidad (por ejemplo, username/email ya registrados).
 */
public class DuplicateResourceException extends RuntimeException {

    public DuplicateResourceException(String message) {
        super(message);
    }
}
