package edu.calidadsoftware.taskmanager.task;

import edu.calidadsoftware.taskmanager.common.ResourceNotFoundException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import javax.validation.Validation;
import javax.validation.Validator;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Pruebas unitarias de TaskService con Mockito.
 *
 * Objetivo: validar casos exitosos, casos límite (título vacío/nulo) e ids inexistentes.
 */
@DisplayName("TaskService")
class TaskServiceTest {

    private TaskRepository taskRepository;
    private Validator validator;
    private TaskService taskService;

    @BeforeEach
    void setUp() {
        taskRepository = Mockito.mock(TaskRepository.class);
        validator = Validation.buildDefaultValidatorFactory().getValidator();
        taskService = new TaskService(taskRepository, validator);
    }

    @AfterEach
    void tearDown() {
        Mockito.reset(taskRepository);
    }

    private static TaskForm validForm() {
        return TaskForm.builder()
                .title("Write tests")
                .description("Add unit tests with Mockito")
                .status(TaskStatus.PENDING)
                .priority(TaskPriority.MEDIUM)
                .build();
    }

    @Nested
    @DisplayName("createTask")
    class CreateTask {

        @Test
        @DisplayName("Crea una tarea válida y la persiste")
        void createTask_success() {
            TaskForm form = validForm();

            when(taskRepository.save(any(Task.class))).thenAnswer(inv -> {
                Task t = inv.getArgument(0);
                t.setId(1L);
                return t;
            });

            Task created = taskService.createTask(form);

            assertNotNull(created);
            assertEquals(1L, created.getId());
            assertEquals("Write tests", created.getTitle());
            assertEquals(TaskStatus.PENDING, created.getStatus());
            assertEquals(TaskPriority.MEDIUM, created.getPriority());
            verify(taskRepository, times(1)).save(any(Task.class));
        }

        @Test
        @DisplayName("Falla si el título es null")
        void createTask_nullTitle_throws() {
            TaskForm form = validForm();
            form.setTitle(null);

            assertThrows(IllegalArgumentException.class, () -> taskService.createTask(form));
            verify(taskRepository, never()).save(any(Task.class));
        }

        @Test
        @DisplayName("Falla si el título está vacío")
        void createTask_blankTitle_throws() {
            TaskForm form = validForm();
            form.setTitle("  ");

            assertThrows(IllegalArgumentException.class, () -> taskService.createTask(form));
            verify(taskRepository, never()).save(any(Task.class));
        }

        @Test
        @DisplayName("Falla si el status es null")
        void createTask_nullStatus_throws() {
            TaskForm form = validForm();
            form.setStatus(null);

            assertThrows(IllegalArgumentException.class, () -> taskService.createTask(form));
            verify(taskRepository, never()).save(any(Task.class));
        }
    }

    @Nested
    @DisplayName("updateTask")
    class UpdateTask {

        @Test
        @DisplayName("Actualiza una tarea existente")
        void updateTask_success() {
            Task existing = Task.builder()
                    .id(10L)
                    .title("Old title")
                    .description("Old desc")
                    .status(TaskStatus.PENDING)
                    .priority(TaskPriority.LOW)
                    .build();

            when(taskRepository.findById(10L)).thenReturn(Optional.of(existing));
            when(taskRepository.save(any(Task.class))).thenAnswer(inv -> inv.getArgument(0));

            TaskForm form = validForm();
            form.setTitle("New title");
            form.setStatus(TaskStatus.IN_PROGRESS);
            form.setPriority(TaskPriority.HIGH);

            Task updated = taskService.updateTask(10L, form);

            assertEquals(10L, updated.getId());
            assertEquals("New title", updated.getTitle());
            assertEquals(TaskStatus.IN_PROGRESS, updated.getStatus());
            assertEquals(TaskPriority.HIGH, updated.getPriority());
            verify(taskRepository, times(1)).findById(10L);
            verify(taskRepository, times(1)).save(any(Task.class));
        }

        @Test
        @DisplayName("Falla si el id no existe")
        void updateTask_nonexistentId_throws() {
            when(taskRepository.findById(999L)).thenReturn(Optional.empty());

            assertThrows(ResourceNotFoundException.class, () -> taskService.updateTask(999L, validForm()));
            verify(taskRepository, times(1)).findById(999L);
            verify(taskRepository, never()).save(any(Task.class));
        }

        @Test
        @DisplayName("Falla si el título es inválido (sin tocar el repositorio)")
        void updateTask_invalidTitle_throwsBeforeRepository() {
            TaskForm form = validForm();
            form.setTitle("");

            assertThrows(IllegalArgumentException.class, () -> taskService.updateTask(10L, form));
            verify(taskRepository, never()).findById(any());
            verify(taskRepository, never()).save(any(Task.class));
        }
    }

    @Nested
    @DisplayName("deleteTask")
    class DeleteTask {

        @Test
        @DisplayName("Elimina una tarea existente")
        void deleteTask_success() {
            Task existing = Task.builder()
                    .id(20L)
                    .title("To delete")
                    .status(TaskStatus.PENDING)
                    .priority(TaskPriority.LOW)
                    .build();

            when(taskRepository.findById(20L)).thenReturn(Optional.of(existing));

            taskService.deleteTask(20L);

            verify(taskRepository, times(1)).findById(20L);
            verify(taskRepository, times(1)).delete(eq(existing));
        }

        @Test
        @DisplayName("Falla si el id no existe")
        void deleteTask_nonexistent_throws() {
            when(taskRepository.findById(404L)).thenReturn(Optional.empty());

            assertThrows(ResourceNotFoundException.class, () -> taskService.deleteTask(404L));
            verify(taskRepository, times(1)).findById(404L);
            verify(taskRepository, never()).delete(any(Task.class));
        }
    }

    @Nested
    @DisplayName("findById")
    class FindById {

        @Test
        @DisplayName("Devuelve una tarea existente")
        void findById_success() {
            Task existing = Task.builder()
                    .id(30L)
                    .title("Exists")
                    .status(TaskStatus.COMPLETED)
                    .priority(TaskPriority.MEDIUM)
                    .build();

            when(taskRepository.findById(30L)).thenReturn(Optional.of(existing));

            Task found = taskService.findById(30L);
            assertEquals(30L, found.getId());
            verify(taskRepository, times(1)).findById(30L);
        }

        @Test
        @DisplayName("Lanza excepción si no existe")
        void findById_notFound() {
            when(taskRepository.findById(31L)).thenReturn(Optional.empty());

            assertThrows(ResourceNotFoundException.class, () -> taskService.findById(31L));
            verify(taskRepository, times(1)).findById(31L);
        }
    }

    @Nested
    @DisplayName("findAll / findByStatus")
    class FindQueries {

        @Test
        @DisplayName("Lista todas las tareas")
        void findAll_success() {
            when(taskRepository.findAll()).thenReturn(Collections.emptyList());

            List<Task> all = taskService.findAll();
            assertNotNull(all);
            verify(taskRepository, times(1)).findAll();
        }

        @Test
        @DisplayName("Filtra por estado")
        void findByStatus_success() {
            List<Task> tasks = Arrays.asList(
                    Task.builder().id(1L).title("A").status(TaskStatus.PENDING).priority(TaskPriority.LOW).build(),
                    Task.builder().id(2L).title("B").status(TaskStatus.PENDING).priority(TaskPriority.HIGH).build()
            );
            when(taskRepository.findByStatus(TaskStatus.PENDING)).thenReturn(tasks);

            List<Task> pending = taskService.findByStatus(TaskStatus.PENDING);
            assertEquals(2, pending.size());
            verify(taskRepository, times(1)).findByStatus(TaskStatus.PENDING);
        }

        @Test
        @DisplayName("Filtra por estado y devuelve vacío cuando no hay coincidencias")
        void findByStatus_empty() {
            when(taskRepository.findByStatus(TaskStatus.COMPLETED)).thenReturn(Collections.emptyList());

            List<Task> completed = taskService.findByStatus(TaskStatus.COMPLETED);
            assertNotNull(completed);
            assertEquals(0, completed.size());
            verify(taskRepository, times(1)).findByStatus(TaskStatus.COMPLETED);
        }
    }
}
