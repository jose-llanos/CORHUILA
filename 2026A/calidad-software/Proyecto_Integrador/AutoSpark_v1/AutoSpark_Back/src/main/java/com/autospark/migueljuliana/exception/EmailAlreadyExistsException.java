package com.autospark.migueljuliana.exception;

public class EmailAlreadyExistsException extends RuntimeException {

    public EmailAlreadyExistsException(String email) {
        super("Email " + email + " is already registered");
    }
}