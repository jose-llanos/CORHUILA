package edu.calidadsoftware.taskmanager.common;

import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;

import javax.servlet.http.HttpServletRequest;

/**
 * Manejo centralizado de excepciones para controladores MVC (Thymeleaf).
 *
 * Permite mostrar páginas de error personalizadas en lugar de stacktraces.
 */
@ControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(ResourceNotFoundException.class)
    public String handleNotFound(ResourceNotFoundException ex, Model model, HttpServletRequest request) {
        model.addAttribute("path", request.getRequestURI());
        model.addAttribute("message", ex.getMessage());
        return "error/404";
    }

    @ExceptionHandler(DuplicateResourceException.class)
    public String handleDuplicate(DuplicateResourceException ex, Model model, HttpServletRequest request) {
        model.addAttribute("path", request.getRequestURI());
        model.addAttribute("message", ex.getMessage());
        return "error/400";
    }

    @ExceptionHandler(Exception.class)
    public String handleGeneric(Exception ex, Model model, HttpServletRequest request) {
        model.addAttribute("path", request.getRequestURI());
        model.addAttribute("message", ex.getMessage());
        return "error/500";
    }
}
