package com.tasks.app.unit.service;

import com.tasks.app.service.TaskService;
import com.tasks.app.dto.request.AssignTaskRequest;
import com.tasks.app.dto.request.ChangeTaskStatusRequest;
import com.tasks.app.dto.request.CreateTaskRequest;
import com.tasks.app.dto.request.UpdateTaskRequest;
import com.tasks.app.dto.response.TaskResponse;
import com.tasks.app.entity.Project;
import com.tasks.app.entity.Task;
import com.tasks.app.entity.TaskStatus;
import com.tasks.app.entity.User;
import com.tasks.app.exception.ForbiddenException;
import com.tasks.app.exception.ResourceNotFoundException;
import com.tasks.app.repository.ProjectMemberRepository;
import com.tasks.app.repository.ProjectRepository;
import com.tasks.app.repository.TaskRepository;
import com.tasks.app.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/*
 * Pruebas unitarias de TaskService.
 * Se simulan todos los repositorios. No se usa BD real.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("TU-03 — TaskService: Gestión de Tareas")
public class TaskServiceTest {

    @Mock private TaskRepository taskRepository;
    @Mock private ProjectRepository projectRepository;
    @Mock private ProjectMemberRepository projectMemberRepository;
    @Mock private UserRepository userRepository;

    @InjectMocks
    private TaskService taskService;

    // Datos reutilizables
    private User owner;
    private User miembro;
    private User usuarioAjeno;
    private Project proyecto;
    private Task tarea;

    @BeforeEach
    void prepararDatos() {
        owner    = buildUser(1L, "owner",   "owner@mail.com");
        miembro  = buildUser(2L, "miembro", "miembro@mail.com");
        usuarioAjeno = buildUser(3L, "ajeno", "ajeno@mail.com");

        proyecto = Project.builder()
                .id(10L).name("Proyecto Alpha").description("Desc").owner(owner).build();

        tarea = Task.builder()
                .id(100L)
                .title("Tarea de prueba")
                .description("Descripción")
                .status(TaskStatus.PENDING)
                .project(proyecto)
                .createdBy(owner)
                .assignedTo(null)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
    }

    // =========================================================
    // TU03-01 a TU03-05 — Crear tarea
    // =========================================================

    @Test
    @DisplayName("TU03-01: El owner puede crear una tarea en el proyecto")
    void crearTarea_porOwner_persiste() {
        // Dado: el proyecto existe y el solicitante es el owner
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, owner)).thenReturn(false);
        when(taskRepository.save(any(Task.class))).thenReturn(tarea);

        // Cuando: crea la tarea
        TaskResponse respuesta = taskService.createTask(10L, crearPeticionTarea("Tarea de prueba", "Desc"), owner);

        // Entonces: se guarda y retorna la tarea
        assertNotNull(respuesta);
        verify(taskRepository, times(1)).save(any(Task.class));
    }

    @Test
    @DisplayName("TU03-02: Un miembro puede crear una tarea en el proyecto")
    void crearTarea_porMiembro_persiste() {
        // Dado: el solicitante es miembro activo
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, miembro)).thenReturn(true);

        Task tareaDelMiembro = Task.builder()
                .id(101L).title("Tarea miembro").description("Desc")
                .status(TaskStatus.PENDING).project(proyecto).createdBy(miembro)
                .createdAt(LocalDateTime.now()).updatedAt(LocalDateTime.now()).build();
        when(taskRepository.save(any(Task.class))).thenReturn(tareaDelMiembro);

        // Cuando: el miembro crea la tarea
        TaskResponse respuesta = taskService.createTask(10L, crearPeticionTarea("Tarea miembro", "Desc"), miembro);

        // Entonces: la tarea se crea correctamente
        assertNotNull(respuesta);
    }

    @Test
    @DisplayName("TU03-03: Un usuario ajeno NO puede crear tareas en el proyecto")
    void crearTarea_porUsuarioAjeno_lanzaForbiddenException() {
        // Dado: el solicitante no tiene relación con el proyecto
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, usuarioAjeno)).thenReturn(false);

        // Cuando + Entonces: se lanza ForbiddenException
        assertThrows(ForbiddenException.class,
                () -> taskService.createTask(10L, crearPeticionTarea("Intento", "Desc"), usuarioAjeno));
    }

    @Test
    @DisplayName("TU03-04: Una tarea nueva siempre se crea con estado PENDING")
    void crearTarea_estadoInicialEsPending() {
        // Dado: la tarea guardada tiene estado PENDING
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, owner)).thenReturn(false);
        when(taskRepository.save(any(Task.class))).thenReturn(tarea);

        // Cuando: se crea la tarea
        TaskResponse respuesta = taskService.createTask(10L, crearPeticionTarea("Tarea", "Desc"), owner);

        // Entonces: el estado es PENDING
        assertEquals(TaskStatus.PENDING, respuesta.getStatus());
    }

    @Test
    @DisplayName("TU03-05: Una tarea nueva se crea sin ningún responsable asignado")
    void crearTarea_assignedToEsNulo() {
        // Dado: la tarea no tiene asignado
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, owner)).thenReturn(false);
        when(taskRepository.save(any(Task.class))).thenReturn(tarea);

        // Cuando: se crea la tarea
        TaskResponse respuesta = taskService.createTask(10L, crearPeticionTarea("Tarea", "Desc"), owner);

        // Entonces: assignedTo es null (sin responsable)
        assertNull(respuesta.getAssignedTo());
    }

    // =========================================================
    // TU03-10 a TU03-12 — Editar tarea
    // =========================================================

    @Test
    @DisplayName("TU03-10: El owner puede editar el título y descripción de una tarea")
    void editarTarea_porOwner_actualizaDatos() {
        // Dado: el proyecto y la tarea existen, el owner solicita la edición
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, owner)).thenReturn(false);
        when(taskRepository.findById(100L)).thenReturn(Optional.of(tarea));

        Task tareaActualizada = buildTaskCompleta(100L, "Título Nuevo", TaskStatus.PENDING, owner, null);
        when(taskRepository.save(any(Task.class))).thenReturn(tareaActualizada);

        // Cuando: el owner edita
        UpdateTaskRequest peticion = crearPeticionEdicion("Título Nuevo", "Desc nueva");
        TaskResponse respuesta = taskService.updateTask(10L, 100L, peticion, owner);

        // Entonces: la respuesta tiene el nuevo título
        assertNotNull(respuesta);
        verify(taskRepository, times(1)).save(any(Task.class));
    }

    @Test
    @DisplayName("TU03-11: Un miembro puede editar cualquier tarea del proyecto")
    void editarTarea_porMiembro_actualizaDatos() {
        // Dado: el miembro tiene acceso al proyecto
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, miembro)).thenReturn(true);
        when(taskRepository.findById(100L)).thenReturn(Optional.of(tarea));

        Task tareaActualizada = buildTaskCompleta(100L, "Editada por miembro", TaskStatus.PENDING, owner, null);
        when(taskRepository.save(any(Task.class))).thenReturn(tareaActualizada);

        // Cuando: el miembro edita la tarea
        UpdateTaskRequest peticion = crearPeticionEdicion("Editada por miembro", "Desc");
        assertDoesNotThrow(() -> taskService.updateTask(10L, 100L, peticion, miembro));
    }

    @Test
    @DisplayName("TU03-12: Un usuario ajeno NO puede editar tareas del proyecto")
    void editarTarea_porUsuarioAjeno_lanzaForbiddenException() {
        // Dado: el solicitante no tiene relación con el proyecto
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, usuarioAjeno)).thenReturn(false);

        // Cuando + Entonces: se lanza ForbiddenException
        UpdateTaskRequest peticion = crearPeticionEdicion("Intento", "Desc");
        assertThrows(ForbiddenException.class,
                () -> taskService.updateTask(10L, 100L, peticion, usuarioAjeno));
    }

    // =========================================================
    // TU03-13, TU03-14 — Eliminar tarea
    // =========================================================

    @Test
    @DisplayName("TU03-13: El owner puede eliminar una tarea")
    void eliminarTarea_porOwner_llama_delete() {
        // Dado: el proyecto y la tarea existen
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(taskRepository.findById(100L)).thenReturn(Optional.of(tarea));

        // Cuando: el owner elimina la tarea
        taskService.deleteTask(10L, 100L, owner);

        // Entonces: se llamó a delete
        verify(taskRepository, times(1)).delete(tarea);
    }

    @Test
    @DisplayName("TU03-14: Un miembro NO puede eliminar tareas")
    void eliminarTarea_porMiembro_lanzaForbiddenException() {
        // Dado: el solicitante es un miembro, no el owner
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));

        // Cuando + Entonces: se lanza ForbiddenException
        assertThrows(ForbiddenException.class,
                () -> taskService.deleteTask(10L, 100L, miembro));
    }

    // =========================================================
    // TU03-15 a TU03-21 — Cambiar estado de tarea
    // =========================================================

    @Test
    @DisplayName("TU03-15: El owner puede cambiar el estado de PENDING a IN_PROGRESS")
    void cambiarEstado_PENDING_a_IN_PROGRESS_porOwner() {
        // Dado: la tarea está en PENDING
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(taskRepository.findById(100L)).thenReturn(Optional.of(tarea));

        Task tareaActualizada = buildTaskCompleta(100L, "Tarea", TaskStatus.IN_PROGRESS, owner, null);
        when(taskRepository.save(any(Task.class))).thenReturn(tareaActualizada);

        // Cuando: el owner la pasa a IN_PROGRESS
        TaskResponse respuesta = taskService.changeStatus(10L, 100L,
                crearPeticionCambioEstado(TaskStatus.IN_PROGRESS), owner);

        // Entonces: el estado es IN_PROGRESS
        assertEquals(TaskStatus.IN_PROGRESS, respuesta.getStatus());
    }

    @Test
    @DisplayName("TU03-16: El owner puede cambiar el estado de IN_PROGRESS a DONE")
    void cambiarEstado_IN_PROGRESS_a_DONE_porOwner() {
        // Dado: la tarea está en IN_PROGRESS
        tarea.setStatus(TaskStatus.IN_PROGRESS);
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(taskRepository.findById(100L)).thenReturn(Optional.of(tarea));

        Task tareaActualizada = buildTaskCompleta(100L, "Tarea", TaskStatus.DONE, owner, null);
        when(taskRepository.save(any(Task.class))).thenReturn(tareaActualizada);

        // Cuando: el owner la marca como DONE
        TaskResponse respuesta = taskService.changeStatus(10L, 100L,
                crearPeticionCambioEstado(TaskStatus.DONE), owner);

        // Entonces: el estado es DONE
        assertEquals(TaskStatus.DONE, respuesta.getStatus());
    }

    @Test
    @DisplayName("TU03-17: El owner puede revertir el estado de DONE a IN_PROGRESS")
    void cambiarEstado_DONE_a_IN_PROGRESS_porOwner() {
        // Dado: la tarea está en DONE
        tarea.setStatus(TaskStatus.DONE);
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(taskRepository.findById(100L)).thenReturn(Optional.of(tarea));

        Task tareaActualizada = buildTaskCompleta(100L, "Tarea", TaskStatus.IN_PROGRESS, owner, null);
        when(taskRepository.save(any(Task.class))).thenReturn(tareaActualizada);

        // Cuando: el owner revierte
        TaskResponse respuesta = taskService.changeStatus(10L, 100L,
                crearPeticionCambioEstado(TaskStatus.IN_PROGRESS), owner);

        // Entonces: el estado es IN_PROGRESS
        assertEquals(TaskStatus.IN_PROGRESS, respuesta.getStatus());
    }

    @Test
    @DisplayName("TU03-18: El owner puede revertir el estado de IN_PROGRESS a PENDING")
    void cambiarEstado_IN_PROGRESS_a_PENDING_porOwner() {
        // Dado: la tarea está en IN_PROGRESS
        tarea.setStatus(TaskStatus.IN_PROGRESS);
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(taskRepository.findById(100L)).thenReturn(Optional.of(tarea));

        Task tareaActualizada = buildTaskCompleta(100L, "Tarea", TaskStatus.PENDING, owner, null);
        when(taskRepository.save(any(Task.class))).thenReturn(tareaActualizada);

        // Cuando: el owner revierte a PENDING
        TaskResponse respuesta = taskService.changeStatus(10L, 100L,
                crearPeticionCambioEstado(TaskStatus.PENDING), owner);

        // Entonces: el estado es PENDING
        assertEquals(TaskStatus.PENDING, respuesta.getStatus());
    }

    @Test
    @DisplayName("TU03-19: El owner puede cambiar el estado de cualquier tarea, incluso asignadas a otros")
    void cambiarEstado_porOwner_incluidasTareasAjenas() {
        // Dado: la tarea está asignada al miembro, pero quien cambia es el owner
        tarea.setAssignedTo(miembro);
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(taskRepository.findById(100L)).thenReturn(Optional.of(tarea));

        Task tareaActualizada = buildTaskCompleta(100L, "Tarea", TaskStatus.IN_PROGRESS, owner, miembro);
        when(taskRepository.save(any(Task.class))).thenReturn(tareaActualizada);

        // Cuando + Entonces: el owner la cambia sin problema
        assertDoesNotThrow(() -> taskService.changeStatus(10L, 100L,
                crearPeticionCambioEstado(TaskStatus.IN_PROGRESS), owner));
    }

    @Test
    @DisplayName("TU03-20: Un miembro puede cambiar el estado de una tarea asignada a él")
    void cambiarEstado_porMiembro_tareaAsignadaASiMismo_permitido() {
        // Dado: la tarea está asignada al miembro
        tarea.setAssignedTo(miembro);
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, miembro)).thenReturn(true);
        when(taskRepository.findById(100L)).thenReturn(Optional.of(tarea));

        Task tareaActualizada = buildTaskCompleta(100L, "Tarea", TaskStatus.IN_PROGRESS, owner, miembro);
        when(taskRepository.save(any(Task.class))).thenReturn(tareaActualizada);

        // Cuando + Entonces: el miembro la cambia sin error
        assertDoesNotThrow(() -> taskService.changeStatus(10L, 100L,
                crearPeticionCambioEstado(TaskStatus.IN_PROGRESS), miembro));
    }

    @Test
    @DisplayName("TU03-21: Un miembro NO puede cambiar el estado de una tarea asignada a otro")
    void cambiarEstado_porMiembro_tareaAsignadaAOtro_lanzaForbiddenException() {
        // Dado: la tarea está asignada al owner, no al miembro
        tarea.setAssignedTo(owner);
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, miembro)).thenReturn(true);
        when(taskRepository.findById(100L)).thenReturn(Optional.of(tarea));

        // Cuando + Entonces: se lanza ForbiddenException
        assertThrows(ForbiddenException.class,
                () -> taskService.changeStatus(10L, 100L,
                        crearPeticionCambioEstado(TaskStatus.IN_PROGRESS), miembro));
    }

    // =========================================================
    // TU03-22 a TU03-27 — Asignar tarea
    // =========================================================

    @Test
    @DisplayName("TU03-22: El owner puede asignar una tarea a un miembro del proyecto")
    void asignarTarea_aMiembro_actualizaAsignado() {
        // Dado: el miembro tiene membresía activa
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(taskRepository.findById(100L)).thenReturn(Optional.of(tarea));
        when(userRepository.findById(2L)).thenReturn(Optional.of(miembro));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, miembro)).thenReturn(true);

        Task tareaAsignada = buildTaskCompleta(100L, "Tarea", TaskStatus.PENDING, owner, miembro);
        when(taskRepository.save(any(Task.class))).thenReturn(tareaAsignada);

        // Cuando: el owner asigna la tarea al miembro
        AssignTaskRequest peticion = crearPeticionAsignacion(2L);
        TaskResponse respuesta = taskService.assignTask(10L, 100L, peticion, owner);

        // Entonces: el assignedTo es el miembro
        assertNotNull(respuesta.getAssignedTo());
    }

    @Test
    @DisplayName("TU03-23: El owner puede autoasignarse una tarea")
    void asignarTarea_ownerSeAutoAsigna_permitido() {
        // Dado: el asignado tiene el mismo id que el owner
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(taskRepository.findById(100L)).thenReturn(Optional.of(tarea));
        when(userRepository.findById(1L)).thenReturn(Optional.of(owner));
        // El owner no aparece en ProjectMember (su rol es por el campo owner en Project)
        when(projectMemberRepository.existsByProjectAndUser(proyecto, owner)).thenReturn(false);

        Task tareaAsignada = buildTaskCompleta(100L, "Tarea", TaskStatus.PENDING, owner, owner);
        when(taskRepository.save(any(Task.class))).thenReturn(tareaAsignada);

        // Cuando + Entonces: el owner se puede autoasignar sin error
        assertDoesNotThrow(() -> taskService.assignTask(10L, 100L, crearPeticionAsignacion(1L), owner));
    }

    @Test
    @DisplayName("TU03-24: El owner puede desasignar una tarea poniendo assignedTo en null")
    void asignarTarea_conNullDesasigna() {
        // Dado: la petición tiene userId = null (desasignar)
        tarea.setAssignedTo(miembro);
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(taskRepository.findById(100L)).thenReturn(Optional.of(tarea));

        Task tareaDesasignada = buildTaskCompleta(100L, "Tarea", TaskStatus.PENDING, owner, null);
        when(taskRepository.save(any(Task.class))).thenReturn(tareaDesasignada);

        // Cuando: se asigna con userId null
        AssignTaskRequest peticion = crearPeticionAsignacion(null);
        TaskResponse respuesta = taskService.assignTask(10L, 100L, peticion, owner);

        // Entonces: assignedTo queda en null
        assertNull(respuesta.getAssignedTo());
    }

    @Test
    @DisplayName("TU03-25: No se puede asignar a un usuario sin membresía en el proyecto")
    void asignarTarea_aUsuarioSinMembresia_lanzaForbiddenException() {
        // Dado: el usuario existe pero no es owner ni miembro del proyecto
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(taskRepository.findById(100L)).thenReturn(Optional.of(tarea));
        when(userRepository.findById(3L)).thenReturn(Optional.of(usuarioAjeno));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, usuarioAjeno)).thenReturn(false);

        // Cuando + Entonces: se lanza ForbiddenException
        assertThrows(ForbiddenException.class,
                () -> taskService.assignTask(10L, 100L, crearPeticionAsignacion(3L), owner));
    }

    @Test
    @DisplayName("TU03-26: No se puede asignar a un usuario que no existe")
    void asignarTarea_aUsuarioInexistente_lanzaResourceNotFoundException() {
        // Dado: el userId no está registrado en BD
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(taskRepository.findById(100L)).thenReturn(Optional.of(tarea));
        when(userRepository.findById(999L)).thenReturn(Optional.empty());

        // Cuando + Entonces: se lanza ResourceNotFoundException
        assertThrows(ResourceNotFoundException.class,
                () -> taskService.assignTask(10L, 100L, crearPeticionAsignacion(999L), owner));
    }

    @Test
    @DisplayName("TU03-27: Un miembro NO puede asignar tareas, solo el owner puede")
    void asignarTarea_porMiembro_lanzaForbiddenException() {
        // Dado: el solicitante es un miembro
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));

        // Cuando + Entonces: se lanza ForbiddenException
        assertThrows(ForbiddenException.class,
                () -> taskService.assignTask(10L, 100L, crearPeticionAsignacion(2L), miembro));
    }

    // =========================================================
    // TU03-29 a TU03-31 — Listar tareas
    // =========================================================

    @Test
    @DisplayName("TU03-29: El owner puede listar las tareas del proyecto")
    void listarTareas_porOwner_retornaLista() {
        // Dado: el proyecto tiene 2 tareas
        Task tarea2 = buildTaskCompleta(101L, "Tarea 2", TaskStatus.DONE, owner, null);
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, owner)).thenReturn(false);
        when(taskRepository.findAllByProject(proyecto)).thenReturn(List.of(tarea, tarea2));

        // Cuando: el owner lista las tareas
        List<TaskResponse> resultado = taskService.listTasks(10L, owner);

        // Entonces: la lista tiene 2 tareas
        assertEquals(2, resultado.size());
    }

    @Test
    @DisplayName("TU03-30: Un miembro puede listar las tareas del proyecto")
    void listarTareas_porMiembro_retornaLista() {
        // Dado: el miembro tiene acceso al proyecto
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, miembro)).thenReturn(true);
        when(taskRepository.findAllByProject(proyecto)).thenReturn(List.of(tarea));

        // Cuando + Entonces: el miembro lista sin error
        assertDoesNotThrow(() -> taskService.listTasks(10L, miembro));
    }

    @Test
    @DisplayName("TU03-31: Un usuario ajeno NO puede listar las tareas del proyecto")
    void listarTareas_porUsuarioAjeno_lanzaForbiddenException() {
        // Dado: el solicitante no tiene ninguna relación con el proyecto
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, usuarioAjeno)).thenReturn(false);

        // Cuando + Entonces: se lanza ForbiddenException
        assertThrows(ForbiddenException.class,
                () -> taskService.listTasks(10L, usuarioAjeno));
    }

    // =========================================================
    // Métodos de ayuda para construir objetos de prueba
    // =========================================================

    private User buildUser(Long id, String username, String email) {
        return User.builder()
                .id(id).username(username).email(email).password("hash")
                .createdAt(LocalDateTime.now()).build();
    }

    private Task buildTaskCompleta(Long id, String titulo, TaskStatus status, User creadoPor, User asignadoA) {
        return Task.builder()
                .id(id).title(titulo).description("Descripción")
                .status(status).project(proyecto)
                .createdBy(creadoPor).assignedTo(asignadoA)
                .createdAt(LocalDateTime.now()).updatedAt(LocalDateTime.now())
                .build();
    }

    private CreateTaskRequest crearPeticionTarea(String titulo, String descripcion) {
        CreateTaskRequest r = new CreateTaskRequest();
        r.setTitle(titulo);
        r.setDescription(descripcion);
        return r;
    }

    private UpdateTaskRequest crearPeticionEdicion(String titulo, String descripcion) {
        UpdateTaskRequest r = new UpdateTaskRequest();
        r.setTitle(titulo);
        r.setDescription(descripcion);
        return r;
    }

    private ChangeTaskStatusRequest crearPeticionCambioEstado(TaskStatus nuevoEstado) {
        ChangeTaskStatusRequest r = new ChangeTaskStatusRequest();
        r.setStatus(nuevoEstado);
        return r;
    }

    private AssignTaskRequest crearPeticionAsignacion(Long userId) {
        AssignTaskRequest r = new AssignTaskRequest();
        r.setAssignedToUserId(userId);
        return r;
    }
}