package edu.calidadsoftware.taskmanager.task;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.Size;

/**
 * DTO para formularios Thymeleaf (crear/editar tareas).
 *
 * Se separa de la entidad para controlar explícitamente los campos editables desde la UI.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class TaskForm {

    @NotBlank(message = "Title is required")
    @Size(min = 3, max = 120, message = "Title must have between 3 and 120 characters")
    private String title;

    @Size(max = 1000, message = "Description must have at most 1000 characters")
    private String description;

    @NotNull(message = "Status is required")
    private TaskStatus status;

    @NotNull(message = "Priority is required")
    private TaskPriority priority;
}
