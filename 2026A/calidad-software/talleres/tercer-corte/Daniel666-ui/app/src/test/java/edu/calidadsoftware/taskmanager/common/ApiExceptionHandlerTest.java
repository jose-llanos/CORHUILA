package edu.calidadsoftware.taskmanager.common;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.BeanPropertyBindingResult;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pruebas unitarias para ApiExceptionHandler.
 *
 * Se valida el formato estándar de error JSON para facilitar automatización.
 */
@DisplayName("ApiExceptionHandler")
class ApiExceptionHandlerTest {

    private final ApiExceptionHandler handler = new ApiExceptionHandler();

    @Nested
    @DisplayName("Errores de dominio")
    class DomainErrors {

        @Test
        @DisplayName("404 cuando ResourceNotFoundException")
        void notFound() {
            ResponseEntity<Map<String, Object>> response = handler.handleNotFound(new ResourceNotFoundException("x"));
            assertEquals(404, response.getStatusCodeValue());
            assertNotNull(response.getBody());
            assertEquals("x", response.getBody().get("message"));
        }

        @Test
        @DisplayName("409 cuando DuplicateResourceException")
        void conflict() {
            ResponseEntity<Map<String, Object>> response = handler.handleDuplicate(new DuplicateResourceException("dup"));
            assertEquals(409, response.getStatusCodeValue());
            assertNotNull(response.getBody());
            assertEquals("dup", response.getBody().get("message"));
        }
    }

    @Test
    @DisplayName("400 cuando falla validación (MethodArgumentNotValidException)")
    void validation() {
        BeanPropertyBindingResult bindingResult = new BeanPropertyBindingResult(new Object(), "req");
        bindingResult.addError(new FieldError("req", "email", "Email must be valid"));
        MethodArgumentNotValidException ex = new MethodArgumentNotValidException(null, bindingResult);

        ResponseEntity<Map<String, Object>> response = handler.handleValidation(ex);

        assertEquals(400, response.getStatusCodeValue());
        assertNotNull(response.getBody());
        assertEquals("Validation failed", response.getBody().get("message"));
        assertTrue(((Map<?, ?>) response.getBody().get("details")).containsKey("email"));
    }

    @Test
    @DisplayName("500 cuando excepción genérica")
    void generic() {
        ResponseEntity<Map<String, Object>> response = handler.handleGeneric(new RuntimeException("boom"));
        assertEquals(500, response.getStatusCodeValue());
        assertNotNull(response.getBody());
        assertEquals("boom", response.getBody().get("message"));
    }
}

