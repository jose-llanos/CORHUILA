package edu.calidadsoftware.taskmanager.web;

import edu.calidadsoftware.taskmanager.task.Task;
import edu.calidadsoftware.taskmanager.task.TaskForm;
import edu.calidadsoftware.taskmanager.task.TaskPriority;
import edu.calidadsoftware.taskmanager.task.TaskService;
import edu.calidadsoftware.taskmanager.task.TaskStatus;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Pruebas unitarias para TaskController (UI Thymeleaf).
 */
@DisplayName("TaskController")
class TaskControllerTest {

    private static TaskForm validForm() {
        return TaskForm.builder()
                .title("Title")
                .description("Desc")
                .status(TaskStatus.PENDING)
                .priority(TaskPriority.MEDIUM)
                .build();
    }

    @Test
    @DisplayName("GET /tasks lista tareas (sin filtro)")
    void list_all() {
        TaskService service = Mockito.mock(TaskService.class);
        when(service.findAll()).thenReturn(Collections.emptyList());

        TaskController controller = new TaskController(service);
        Model model = Mockito.mock(Model.class);

        String view = controller.list(null, model);

        assertEquals("tasks/list", view);
        verify(model).addAttribute(eq("tasks"), any());
        verify(model).addAttribute(eq("selectedStatus"), eq(null));
        verify(model).addAttribute(eq("statuses"), any());
    }

    @Test
    @DisplayName("GET /tasks filtra por status")
    void list_byStatus() {
        TaskService service = Mockito.mock(TaskService.class);
        when(service.findByStatus(TaskStatus.COMPLETED)).thenReturn(Collections.emptyList());

        TaskController controller = new TaskController(service);
        Model model = Mockito.mock(Model.class);

        String view = controller.list(TaskStatus.COMPLETED, model);

        assertEquals("tasks/list", view);
        verify(model).addAttribute(eq("selectedStatus"), eq(TaskStatus.COMPLETED));
    }

    @Test
    @DisplayName("GET /tasks/new carga formulario con defaults")
    void createForm_defaults() {
        TaskService service = Mockito.mock(TaskService.class);
        TaskController controller = new TaskController(service);

        Model model = Mockito.mock(Model.class);
        String view = controller.createForm(model);

        assertEquals("tasks/form", view);
        verify(model).addAttribute(eq("taskForm"), any(TaskForm.class));
        verify(model).addAttribute(eq("statuses"), any());
        verify(model).addAttribute(eq("priorities"), any());
        verify(model).addAttribute(eq("mode"), eq("create"));
    }

    @Nested
    @DisplayName("POST create/update")
    class CreateUpdate {

        @Test
        @DisplayName("POST /tasks con errores devuelve la vista del formulario")
        void create_validationErrors() {
            TaskService service = Mockito.mock(TaskService.class);
            TaskController controller = new TaskController(service);

            BindingResult binding = Mockito.mock(BindingResult.class);
            when(binding.hasErrors()).thenReturn(true);

            Model model = Mockito.mock(Model.class);
            RedirectAttributes redirect = Mockito.mock(RedirectAttributes.class);

            String view = controller.create(validForm(), binding, model, redirect);

            assertEquals("tasks/form", view);
            verify(service, never()).createTask(any(TaskForm.class));
        }

        @Test
        @DisplayName("POST /tasks exitoso redirige a /tasks")
        void create_success() {
            TaskService service = Mockito.mock(TaskService.class);
            when(service.createTask(any(TaskForm.class))).thenReturn(Task.builder().id(1L).build());

            TaskController controller = new TaskController(service);

            BindingResult binding = Mockito.mock(BindingResult.class);
            when(binding.hasErrors()).thenReturn(false);

            Model model = Mockito.mock(Model.class);
            RedirectAttributes redirect = Mockito.mock(RedirectAttributes.class);

            String view = controller.create(validForm(), binding, model, redirect);

            assertEquals("redirect:/tasks", view);
            verify(service).createTask(any(TaskForm.class));
            verify(redirect).addFlashAttribute(eq("message"), any());
        }

        @Test
        @DisplayName("POST /tasks/{id} con errores devuelve vista de formulario")
        void update_validationErrors() {
            TaskService service = Mockito.mock(TaskService.class);
            TaskController controller = new TaskController(service);

            BindingResult binding = Mockito.mock(BindingResult.class);
            when(binding.hasErrors()).thenReturn(true);

            Model model = Mockito.mock(Model.class);
            RedirectAttributes redirect = Mockito.mock(RedirectAttributes.class);

            String view = controller.update(10L, validForm(), binding, model, redirect);

            assertEquals("tasks/form", view);
            verify(service, never()).updateTask(any(), any());
        }

        @Test
        @DisplayName("POST /tasks/{id} exitoso redirige a /tasks")
        void update_success() {
            TaskService service = Mockito.mock(TaskService.class);
            when(service.updateTask(eq(10L), any(TaskForm.class))).thenReturn(Task.builder().id(10L).build());

            TaskController controller = new TaskController(service);

            BindingResult binding = Mockito.mock(BindingResult.class);
            when(binding.hasErrors()).thenReturn(false);

            Model model = Mockito.mock(Model.class);
            RedirectAttributes redirect = Mockito.mock(RedirectAttributes.class);

            String view = controller.update(10L, validForm(), binding, model, redirect);

            assertEquals("redirect:/tasks", view);
            verify(service).updateTask(eq(10L), any(TaskForm.class));
            verify(redirect).addFlashAttribute(eq("message"), any());
        }
    }

    @Test
    @DisplayName("GET /tasks/{id}/edit carga formulario con datos")
    void editForm_loadsTask() {
        TaskService service = Mockito.mock(TaskService.class);
        when(service.findById(5L)).thenReturn(Task.builder()
                .id(5L)
                .title("t")
                .description("d")
                .status(TaskStatus.IN_PROGRESS)
                .priority(TaskPriority.HIGH)
                .build());

        TaskController controller = new TaskController(service);
        Model model = Mockito.mock(Model.class);

        String view = controller.editForm(5L, model);

        assertEquals("tasks/form", view);
        verify(model).addAttribute(eq("taskId"), eq(5L));
        verify(model).addAttribute(eq("taskForm"), any(TaskForm.class));
        verify(model).addAttribute(eq("mode"), eq("edit"));
    }

    @Test
    @DisplayName("POST /tasks/{id}/delete redirige a /tasks")
    void delete_redirects() {
        TaskService service = Mockito.mock(TaskService.class);
        TaskController controller = new TaskController(service);
        RedirectAttributes redirect = Mockito.mock(RedirectAttributes.class);

        String view = controller.delete(7L, redirect);

        assertEquals("redirect:/tasks", view);
        verify(service).deleteTask(7L);
        verify(redirect).addFlashAttribute(eq("message"), any());
    }
}

