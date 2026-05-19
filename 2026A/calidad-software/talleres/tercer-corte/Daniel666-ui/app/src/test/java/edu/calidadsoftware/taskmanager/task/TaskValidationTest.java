package edu.calidadsoftware.taskmanager.task;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import javax.validation.ConstraintViolation;
import javax.validation.Validation;
import javax.validation.Validator;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pruebas de validación con Bean Validation (javax.validation).
 *
 * Objetivo: asegurar que los DTOs (TaskForm) cumplen las restricciones declaradas.
 */
@DisplayName("Task Bean Validation")
class TaskValidationTest {

    private Validator validator;

    @BeforeEach
    void setUp() {
        validator = Validation.buildDefaultValidatorFactory().getValidator();
    }

    private static TaskForm validForm() {
        return TaskForm.builder()
                .title("Valid title")
                .description("Valid description")
                .status(TaskStatus.PENDING)
                .priority(TaskPriority.MEDIUM)
                .build();
    }

    @Nested
    @DisplayName("title")
    class TitleValidation {

        @Test
        @DisplayName("Título válido pasa la validación")
        void title_valid_ok() {
            Set<ConstraintViolation<TaskForm>> violations = validator.validate(validForm());
            assertTrue(violations.isEmpty());
        }

        @Test
        @DisplayName("Título null falla")
        void title_null_fails() {
            TaskForm form = validForm();
            form.setTitle(null);
            Set<ConstraintViolation<TaskForm>> violations = validator.validate(form);
            assertFalse(violations.isEmpty());
        }

        @Test
        @DisplayName("Título vacío falla")
        void title_blank_fails() {
            TaskForm form = validForm();
            form.setTitle(" ");
            Set<ConstraintViolation<TaskForm>> violations = validator.validate(form);
            assertFalse(violations.isEmpty());
        }

        @Test
        @DisplayName("Título demasiado corto falla")
        void title_tooShort_fails() {
            TaskForm form = validForm();
            form.setTitle("ab");
            Set<ConstraintViolation<TaskForm>> violations = validator.validate(form);
            assertFalse(violations.isEmpty());
        }
    }

    @Nested
    @DisplayName("status/priority")
    class EnumValidation {

        @Test
        @DisplayName("Status null falla")
        void status_null_fails() {
            TaskForm form = validForm();
            form.setStatus(null);
            Set<ConstraintViolation<TaskForm>> violations = validator.validate(form);
            assertFalse(violations.isEmpty());
        }

        @Test
        @DisplayName("Priority null falla")
        void priority_null_fails() {
            TaskForm form = validForm();
            form.setPriority(null);
            Set<ConstraintViolation<TaskForm>> violations = validator.validate(form);
            assertFalse(violations.isEmpty());
        }
    }
}
