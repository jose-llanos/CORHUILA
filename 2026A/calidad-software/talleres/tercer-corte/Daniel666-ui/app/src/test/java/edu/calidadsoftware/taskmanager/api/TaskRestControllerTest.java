package edu.calidadsoftware.taskmanager.api;

import edu.calidadsoftware.taskmanager.task.Task;
import edu.calidadsoftware.taskmanager.task.TaskForm;
import edu.calidadsoftware.taskmanager.task.TaskPriority;
import edu.calidadsoftware.taskmanager.task.TaskService;
import edu.calidadsoftware.taskmanager.task.TaskStatus;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.http.ResponseEntity;

import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;

/**
 * Pruebas unitarias para TaskRestController.
 *
 * Se testea el controlador aislado usando un mock de TaskService.
 */
@DisplayName("TaskRestController")
class TaskRestControllerTest {

    private static TaskForm form() {
        return TaskForm.builder()
                .title("t")
                .description("d")
                .status(TaskStatus.PENDING)
                .priority(TaskPriority.MEDIUM)
                .build();
    }

    @Nested
    @DisplayName("list")
    class ListTests {

        @Test
        @DisplayName("Lista sin filtro usa findAll")
        void list_all() {
            TaskService service = Mockito.mock(TaskService.class);
            when(service.findAll()).thenReturn(Collections.emptyList());

            TaskRestController controller = new TaskRestController(service);
            List<Task> result = controller.list(null);
            assertNotNull(result);
            assertEquals(0, result.size());
        }

        @Test
        @DisplayName("Lista con filtro usa findByStatus")
        void list_byStatus() {
            TaskService service = Mockito.mock(TaskService.class);
            when(service.findByStatus(TaskStatus.COMPLETED)).thenReturn(Collections.emptyList());

            TaskRestController controller = new TaskRestController(service);
            List<Task> result = controller.list(TaskStatus.COMPLETED);
            assertNotNull(result);
        }
    }

    @Test
    @DisplayName("create retorna 201 y body con task creada")
    void create_created() {
        TaskService service = Mockito.mock(TaskService.class);
        Task created = Task.builder().id(1L).title("t").status(TaskStatus.PENDING).priority(TaskPriority.MEDIUM).build();
        when(service.createTask(any(TaskForm.class))).thenReturn(created);

        TaskRestController controller = new TaskRestController(service);
        ResponseEntity<Task> response = controller.create(form());

        assertEquals(201, response.getStatusCodeValue());
        assertNotNull(response.getBody());
        assertEquals(1L, response.getBody().getId());
    }

    @Test
    @DisplayName("findById delega a service")
    void findById_ok() {
        TaskService service = Mockito.mock(TaskService.class);
        when(service.findById(10L)).thenReturn(Task.builder().id(10L).title("x").status(TaskStatus.PENDING).priority(TaskPriority.LOW).build());

        TaskRestController controller = new TaskRestController(service);
        Task found = controller.findById(10L);
        assertEquals(10L, found.getId());
    }

    @Test
    @DisplayName("update delega a service")
    void update_ok() {
        TaskService service = Mockito.mock(TaskService.class);
        when(service.updateTask(eq(2L), any(TaskForm.class))).thenReturn(Task.builder().id(2L).title("u").status(TaskStatus.IN_PROGRESS).priority(TaskPriority.HIGH).build());

        TaskRestController controller = new TaskRestController(service);
        Task updated = controller.update(2L, form());
        assertEquals(2L, updated.getId());
        assertEquals(TaskStatus.IN_PROGRESS, updated.getStatus());
    }

    @Test
    @DisplayName("delete retorna 204")
    void delete_noContent() {
        TaskService service = Mockito.mock(TaskService.class);
        TaskRestController controller = new TaskRestController(service);
        ResponseEntity<Void> response = controller.delete(99L);
        assertEquals(204, response.getStatusCodeValue());
    }
}

