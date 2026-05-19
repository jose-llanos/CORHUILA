package edu.calidadsoftware.taskmanager.common;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.ui.Model;

import javax.servlet.http.HttpServletRequest;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Pruebas unitarias para GlobalExceptionHandler (MVC).
 */
@DisplayName("GlobalExceptionHandler")
class GlobalExceptionHandlerTest {

    private final GlobalExceptionHandler handler = new GlobalExceptionHandler();

    @Test
    @DisplayName("Devuelve vista 404 y setea atributos")
    void handleNotFound() {
        Model model = Mockito.mock(Model.class);
        HttpServletRequest request = Mockito.mock(HttpServletRequest.class);
        when(request.getRequestURI()).thenReturn("/tasks/999");

        String view = handler.handleNotFound(new ResourceNotFoundException("not found"), model, request);

        assertEquals("error/404", view);
        verify(model).addAttribute("path", "/tasks/999");
        verify(model).addAttribute("message", "not found");
    }

    @Test
    @DisplayName("Devuelve vista 400 y setea atributos")
    void handleDuplicate() {
        Model model = Mockito.mock(Model.class);
        HttpServletRequest request = Mockito.mock(HttpServletRequest.class);
        when(request.getRequestURI()).thenReturn("/api/users");

        String view = handler.handleDuplicate(new DuplicateResourceException("dup"), model, request);

        assertEquals("error/400", view);
        verify(model).addAttribute("path", "/api/users");
        verify(model).addAttribute("message", "dup");
    }

    @Test
    @DisplayName("Devuelve vista 500 y setea atributos")
    void handleGeneric() {
        Model model = Mockito.mock(Model.class);
        HttpServletRequest request = Mockito.mock(HttpServletRequest.class);
        when(request.getRequestURI()).thenReturn("/x");

        String view = handler.handleGeneric(new RuntimeException("boom"), model, request);

        assertEquals("error/500", view);
        verify(model).addAttribute("path", "/x");
        verify(model).addAttribute("message", "boom");
    }
}

