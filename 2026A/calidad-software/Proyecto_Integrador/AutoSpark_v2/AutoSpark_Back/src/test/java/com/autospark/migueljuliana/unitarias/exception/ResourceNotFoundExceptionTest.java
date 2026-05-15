package com.autospark.migueljuliana.unitarias.exception;

import com.autospark.migueljuliana.exception.ResourceNotFoundException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ResourceNotFoundExceptionTest {

    @Test
    void constructorConResourceEIdDebeCrearMensajeCorrecto() {
        ResourceNotFoundException exception =
                new ResourceNotFoundException("User", 1L);

        assertEquals(
                "User with id 1 not found",
                exception.getMessage()
        );
    }

    @Test
    void constructorConMensajeDebeCrearMensajeCorrecto() {
        ResourceNotFoundException exception =
                new ResourceNotFoundException("Custom error");

        assertEquals(
                "Custom error",
                exception.getMessage()
        );
    }
}