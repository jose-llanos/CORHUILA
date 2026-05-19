package edu.calidadsoftware.taskmanager.task;

import edu.calidadsoftware.taskmanager.common.ResourceNotFoundException;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.validation.ConstraintViolation;
import javax.validation.Validator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Capa de servicio para Task.
 *
 * Centraliza reglas de negocio y acceso a datos. Esto facilita pruebas unitarias con mocks.
 */
@Service
@RequiredArgsConstructor
public class TaskService {

    private final TaskRepository taskRepository;
    private final Validator validator;

    @Transactional
    public Task createTask(TaskForm form) {
        validateOrThrow(form);
        Task task = Task.builder()
                .title(form.getTitle())
                .description(form.getDescription())
                .status(form.getStatus())
                .priority(form.getPriority())
                .build();
        return taskRepository.save(task);
    }

    @Transactional
    public Task updateTask(Long id, TaskForm form) {
        validateOrThrow(form);
        Task existing = findById(id);
        existing.setTitle(form.getTitle());
        existing.setDescription(form.getDescription());
        existing.setStatus(form.getStatus());
        existing.setPriority(form.getPriority());
        return taskRepository.save(existing);
    }

    @Transactional
    public void deleteTask(Long id) {
        Task existing = findById(id);
        taskRepository.delete(existing);
    }

    @Transactional(readOnly = true)
    public Task findById(Long id) {
        return taskRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Task not found: id=" + id));
    }

    @Transactional(readOnly = true)
    public List<Task> findAll() {
        return taskRepository.findAll();
    }

    @Transactional(readOnly = true)
    public List<Task> findByStatus(TaskStatus status) {
        return taskRepository.findByStatus(status);
    }

    private void validateOrThrow(Object target) {
        Set<ConstraintViolation<Object>> violations = validator.validate(target);
        if (!violations.isEmpty()) {
            String details = violations.stream()
                    .map(v -> v.getPropertyPath() + ": " + v.getMessage())
                    .collect(Collectors.joining(", "));
            throw new IllegalArgumentException("Validation failed: " + details);
        }
    }
}
