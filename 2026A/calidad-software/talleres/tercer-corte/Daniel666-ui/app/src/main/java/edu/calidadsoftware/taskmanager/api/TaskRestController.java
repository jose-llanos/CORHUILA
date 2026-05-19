package edu.calidadsoftware.taskmanager.api;

import edu.calidadsoftware.taskmanager.task.Task;
import edu.calidadsoftware.taskmanager.task.TaskForm;
import edu.calidadsoftware.taskmanager.task.TaskService;
import edu.calidadsoftware.taskmanager.task.TaskStatus;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import javax.validation.Valid;
import java.util.List;

/**
 * API REST para tareas.
 *
 * Rutas:
 * - GET /api/tasks?status=PENDING
 * - POST /api/tasks
 * - GET /api/tasks/{id}
 * - PUT /api/tasks/{id}
 * - DELETE /api/tasks/{id}
 */
@RestController
@RequestMapping("/api/tasks")
@RequiredArgsConstructor
@Validated
public class TaskRestController {

    private final TaskService taskService;

    @GetMapping
    public List<Task> list(@RequestParam(value = "status", required = false) TaskStatus status) {
        return (status == null) ? taskService.findAll() : taskService.findByStatus(status);
    }

    @PostMapping
    public ResponseEntity<Task> create(@Valid @RequestBody TaskForm form) {
        Task created = taskService.createTask(form);
        return ResponseEntity.status(HttpStatus.CREATED).body(created);
    }

    @GetMapping("/{id}")
    public Task findById(@PathVariable Long id) {
        return taskService.findById(id);
    }

    @PutMapping("/{id}")
    public Task update(@PathVariable Long id, @Valid @RequestBody TaskForm form) {
        return taskService.updateTask(id, form);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        taskService.deleteTask(id);
        return ResponseEntity.noContent().build();
    }
}
