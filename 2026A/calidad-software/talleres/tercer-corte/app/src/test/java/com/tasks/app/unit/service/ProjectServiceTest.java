package com.tasks.app.unit.service;

import com.tasks.app.service.ProjectService;
import com.tasks.app.dto.request.CreateProjectRequest;
import com.tasks.app.dto.request.InviteMemberRequest;
import com.tasks.app.dto.request.UpdateProjectRequest;
import com.tasks.app.dto.response.MemberResponse;
import com.tasks.app.dto.response.ProjectDetailResponse;
import com.tasks.app.dto.response.ProjectResponse;
import com.tasks.app.entity.Project;
import com.tasks.app.entity.ProjectMember;
import com.tasks.app.entity.Task;
import com.tasks.app.entity.TaskStatus;
import com.tasks.app.entity.User;
import com.tasks.app.exception.ConflictException;
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
 * Pruebas unitarias de ProjectService.
 * Se simulan (mock) todos los repositorios. No se usa BD real.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("TU-02 — ProjectService: Gestión de Proyectos")
public class ProjectServiceTest {

    @Mock private ProjectRepository projectRepository;
    @Mock private ProjectMemberRepository projectMemberRepository;
    @Mock private UserRepository userRepository;
    @Mock private TaskRepository taskRepository;

    @InjectMocks
    private ProjectService projectService;

    // Datos de prueba reutilizables
    private User owner;
    private User miembro;
    private User usuarioAjeno;
    private Project proyecto;
    private ProjectMember membresia;

    @BeforeEach
    void prepararDatos() {
        owner = buildUser(1L, "owner", "owner@mail.com");
        miembro = buildUser(2L, "miembro", "miembro@mail.com");
        usuarioAjeno = buildUser(3L, "ajeno", "ajeno@mail.com");

        proyecto = Project.builder()
                .id(10L)
                .name("Proyecto Alpha")
                .description("Descripción de prueba")
                .owner(owner)
                .build();

        membresia = ProjectMember.builder()
                .id(1L)
                .project(proyecto)
                .user(miembro)
                .joinedAt(LocalDateTime.now())
                .build();
    }

    // =========================================================
    // TU02-01, 03, 04 — Crear proyecto
    // =========================================================

    @Test
    @DisplayName("TU02-01: Crear proyecto exitoso — lo persiste y retorna sus datos")
    void crearProyecto_datosValidos_guardaYRetorna() {
        // Dado: no hay proyecto con ese nombre para ese owner
        CreateProjectRequest peticion = crearPeticionProyecto("Proyecto Alpha", "Descripción");
        when(projectRepository.existsByNameAndOwner("Proyecto Alpha", owner)).thenReturn(false);
        when(projectRepository.save(any(Project.class))).thenReturn(proyecto);

        // Cuando: el owner crea el proyecto
        ProjectResponse respuesta = projectService.createProject(peticion, owner);

        // Entonces: el proyecto se guarda y la respuesta contiene el nombre
        assertNotNull(respuesta);
        assertEquals("Proyecto Alpha", respuesta.getName());
        verify(projectRepository, times(1)).save(any(Project.class));
    }

    @Test
    @DisplayName("TU02-03: Crear proyecto falla si el mismo owner ya tiene ese nombre")
    void crearProyecto_nombreDuplicadoMismoOwner_lanzaConflictException() {
        // Dado: ya existe un proyecto con ese nombre para ese owner
        CreateProjectRequest peticion = crearPeticionProyecto("Proyecto Alpha", "Descripción");
        when(projectRepository.existsByNameAndOwner("Proyecto Alpha", owner)).thenReturn(true);

        // Cuando + Entonces: se lanza ConflictException
        assertThrows(ConflictException.class, () -> projectService.createProject(peticion, owner));
        verify(projectRepository, never()).save(any());
    }

    @Test
    @DisplayName("TU02-04: Crear proyecto con mismo nombre pero diferente owner está permitido")
    void crearProyecto_mismoNombreDiferenteOwner_creaCorrectamente() {
        // Dado: para el 'usuarioAjeno' ese nombre NO existe
        CreateProjectRequest peticion = crearPeticionProyecto("Proyecto Alpha", "Descripción");
        Project proyectoDiferenteOwner = Project.builder()
                .id(99L).name("Proyecto Alpha").description("Desc").owner(usuarioAjeno).build();
        when(projectRepository.existsByNameAndOwner("Proyecto Alpha", usuarioAjeno)).thenReturn(false);
        when(projectRepository.save(any(Project.class))).thenReturn(proyectoDiferenteOwner);

        // Cuando: otro usuario crea un proyecto con el mismo nombre
        ProjectResponse respuesta = projectService.createProject(peticion, usuarioAjeno);

        // Entonces: el proyecto se crea sin error
        assertNotNull(respuesta);
    }

    // =========================================================
    // TU02-07 a TU02-10 — Listar proyectos
    // =========================================================

    @Test
    @DisplayName("TU02-07: Listar proyectos incluye los proyectos donde el usuario es owner")
    void listarProyectos_esOwner_incluyeSusProyectos() {
        // Dado: el owner tiene 2 proyectos propios
        Project p2 = Project.builder().id(11L).name("Proyecto Beta").owner(owner).build();
        when(projectRepository.findAllAccessibleByUser(owner)).thenReturn(List.of(proyecto, p2));

        // Cuando: lista sus proyectos
        List<ProjectResponse> resultado = projectService.listProjects(owner);

        // Entonces: la lista contiene exactamente 2 proyectos
        assertEquals(2, resultado.size());
    }

    @Test
    @DisplayName("TU02-08: Listar proyectos incluye los proyectos donde el usuario es miembro")
    void listarProyectos_esMiembro_incluyeProyectosDeSuMembresia() {
        // Dado: el miembro tiene acceso a 1 proyecto ajeno
        when(projectRepository.findAllAccessibleByUser(miembro)).thenReturn(List.of(proyecto));

        // Cuando: el miembro lista sus proyectos accesibles
        List<ProjectResponse> resultado = projectService.listProjects(miembro);

        // Entonces: aparece el proyecto al que tiene membresía
        assertEquals(1, resultado.size());
    }

    @Test
    @DisplayName("TU02-09: Listar combina proyectos propios + de membresía sin duplicados")
    void listarProyectos_ownerYMiembro_combinaSinDuplicados() {
        // Dado: el usuario es owner de 1 y miembro de otro (2 en total)
        Project proyectoComoMiembro = Project.builder().id(20L).name("Proyecto Ajeno").owner(usuarioAjeno).build();
        when(projectRepository.findAllAccessibleByUser(owner)).thenReturn(List.of(proyecto, proyectoComoMiembro));

        // Cuando: lista
        List<ProjectResponse> resultado = projectService.listProjects(owner);

        // Entonces: son exactamente 2, sin repetidos
        assertEquals(2, resultado.size());
    }

    @Test
    @DisplayName("TU02-10: Listar retorna lista vacía si no tiene proyectos ni membresías")
    void listarProyectos_sinProyectos_retornaListaVacia() {
        // Dado: el usuario no tiene ningún proyecto ni membresía
        when(projectRepository.findAllAccessibleByUser(usuarioAjeno)).thenReturn(List.of());

        // Cuando: lista
        List<ProjectResponse> resultado = projectService.listProjects(usuarioAjeno);

        // Entonces: la lista está vacía (no es null)
        assertNotNull(resultado);
        assertTrue(resultado.isEmpty());
    }

    // =========================================================
    // TU02-11 a TU02-14 — Ver detalle del proyecto
    // =========================================================

    @Test
    @DisplayName("TU02-11: Ver detalle agrupa correctamente las tareas por estado")
    void verDetalle_conTareasEnDiferentesEstados_agrupa() {
        // Dado: un proyecto con 2 PENDING, 1 IN_PROGRESS y 3 DONE
        List<Task> tareas = List.of(
                buildTask(1L, TaskStatus.PENDING),
                buildTask(2L, TaskStatus.PENDING),
                buildTask(3L, TaskStatus.IN_PROGRESS),
                buildTask(4L, TaskStatus.DONE),
                buildTask(5L, TaskStatus.DONE),
                buildTask(6L, TaskStatus.DONE)
        );
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, owner)).thenReturn(false);
        when(taskRepository.findAllByProject(proyecto)).thenReturn(tareas);

        // Cuando: el owner consulta el detalle
        ProjectDetailResponse detalle = projectService.getProjectDetail(10L, owner);

        // Entonces: cada grupo tiene el tamaño correcto
        assertEquals(2, detalle.getPending().size());
        assertEquals(1, detalle.getInProgress().size());
        assertEquals(3, detalle.getDone().size());
    }

    @Test
    @DisplayName("TU02-12: El owner puede ver el detalle de su propio proyecto")
    void verDetalle_porOwner_retornaDetalle() {
        // Dado: el proyecto existe y el solicitante es el owner
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, owner)).thenReturn(false);
        when(taskRepository.findAllByProject(proyecto)).thenReturn(List.of());

        // Cuando + Entonces: no se lanza ninguna excepción
        assertDoesNotThrow(() -> projectService.getProjectDetail(10L, owner));
    }

    @Test
    @DisplayName("TU02-13: Un miembro puede ver el detalle del proyecto")
    void verDetalle_porMiembro_retornaDetalle() {
        // Dado: el proyecto existe y el solicitante tiene membresía activa
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, miembro)).thenReturn(true);
        when(taskRepository.findAllByProject(proyecto)).thenReturn(List.of());

        // Cuando + Entonces: el miembro accede sin error
        assertDoesNotThrow(() -> projectService.getProjectDetail(10L, miembro));
    }

    @Test
    @DisplayName("TU02-14: Un usuario ajeno NO puede ver el detalle del proyecto")
    void verDetalle_porUsuarioAjeno_lanzaForbiddenException() {
        // Dado: el proyecto existe pero el solicitante no es owner ni miembro
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, usuarioAjeno)).thenReturn(false);

        // Cuando + Entonces: se lanza ForbiddenException
        assertThrows(ForbiddenException.class, () -> projectService.getProjectDetail(10L, usuarioAjeno));
    }

    // =========================================================
    // TU02-15, TU02-16 — Editar proyecto
    // =========================================================

    @Test
    @DisplayName("TU02-15: El owner puede editar nombre y descripción del proyecto")
    void editarProyecto_porOwner_actualizaDatos() {
        // Dado: el proyecto existe, el solicitante es el owner, nuevo nombre no duplicado
        UpdateProjectRequest peticion = crearPeticionActualizacion("Proyecto Renombrado", "Nueva desc");
        Project proyectoActualizado = Project.builder()
                .id(10L).name("Proyecto Renombrado").description("Nueva desc").owner(owner).build();
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectRepository.existsByNameAndOwner("Proyecto Renombrado", owner)).thenReturn(false);
        when(projectRepository.save(any(Project.class))).thenReturn(proyectoActualizado);

        // Cuando: el owner edita el proyecto
        ProjectResponse respuesta = projectService.updateProject(10L, peticion, owner);

        // Entonces: el nombre fue actualizado
        assertEquals("Proyecto Renombrado", respuesta.getName());
    }

    @Test
    @DisplayName("TU02-16: Un miembro NO puede editar el proyecto")
    void editarProyecto_porMiembro_lanzaForbiddenException() {
        // Dado: el proyecto existe pero el solicitante es miembro, no owner
        UpdateProjectRequest peticion = crearPeticionActualizacion("Intento Miembro", "Desc");
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));

        // Cuando + Entonces: se lanza ForbiddenException
        assertThrows(ForbiddenException.class, () -> projectService.updateProject(10L, peticion, miembro));
    }

    // =========================================================
    // TU02-17, TU02-20 — Eliminar proyecto
    // =========================================================

    @Test
    @DisplayName("TU02-17: El owner puede eliminar el proyecto")
    void eliminarProyecto_porOwner_llama_delete() {
        // Dado: el proyecto existe y el solicitante es el owner
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));

        // Cuando: el owner elimina el proyecto
        projectService.deleteProject(10L, owner);

        // Entonces: se llamó a delete (la cascada de BD borrará tareas y membresías)
        verify(projectRepository, times(1)).delete(proyecto);
    }

    @Test
    @DisplayName("TU02-20: Un miembro NO puede eliminar el proyecto")
    void eliminarProyecto_porMiembro_lanzaForbiddenException() {
        // Dado: el proyecto existe pero el solicitante es un miembro
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));

        // Cuando + Entonces: se lanza ForbiddenException
        assertThrows(ForbiddenException.class, () -> projectService.deleteProject(10L, miembro));
    }

    // =========================================================
    // TU02-21 a TU02-25 — Invitar miembro
    // =========================================================

    @Test
    @DisplayName("TU02-21: El owner puede invitar a un usuario existente no miembro")
    void invitarMiembro_datosValidos_creaMembresia() {
        // Dado: el usuario existe, no es owner, no es miembro ya
        InviteMemberRequest peticion = crearPeticionInvitacion("miembro");
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(userRepository.findByUsername("miembro")).thenReturn(Optional.of(miembro));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, miembro)).thenReturn(false);
        when(projectMemberRepository.save(any(ProjectMember.class))).thenReturn(membresia);

        // Cuando: el owner invita
        MemberResponse respuesta = projectService.inviteMember(10L, peticion, owner);

        // Entonces: se crea la membresía y se retorna el nuevo miembro
        assertNotNull(respuesta);
        verify(projectMemberRepository, times(1)).save(any(ProjectMember.class));
    }

    @Test
    @DisplayName("TU02-22: Invitar a un usuario que no existe lanza ResourceNotFoundException")
    void invitarMiembro_usuarioInexistente_lanzaResourceNotFoundException() {
        // Dado: el username no está registrado
        InviteMemberRequest peticion = crearPeticionInvitacion("nadie");
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(userRepository.findByUsername("nadie")).thenReturn(Optional.empty());

        // Cuando + Entonces: se lanza ResourceNotFoundException
        assertThrows(ResourceNotFoundException.class,
                () -> projectService.inviteMember(10L, peticion, owner));
    }

    @Test
    @DisplayName("TU02-23: El owner no puede invitarse a sí mismo al proyecto")
    void invitarMiembro_ownerSeAutoInvita_lanzaConflictException() {
        // Dado: el invitado tiene el mismo id que el owner
        InviteMemberRequest peticion = crearPeticionInvitacion("owner");
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(userRepository.findByUsername("owner")).thenReturn(Optional.of(owner));

        // Cuando + Entonces: se lanza ConflictException
        assertThrows(ConflictException.class,
                () -> projectService.inviteMember(10L, peticion, owner));
    }

    @Test
    @DisplayName("TU02-24: Invitar a un usuario que ya es miembro lanza ConflictException")
    void invitarMiembro_yaEsMiembro_lanzaConflictException() {
        // Dado: el usuario ya tiene membresía activa
        InviteMemberRequest peticion = crearPeticionInvitacion("miembro");
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(userRepository.findByUsername("miembro")).thenReturn(Optional.of(miembro));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, miembro)).thenReturn(true);

        // Cuando + Entonces: se lanza ConflictException
        assertThrows(ConflictException.class,
                () -> projectService.inviteMember(10L, peticion, owner));
    }

    @Test
    @DisplayName("TU02-25: Solo el owner puede invitar miembros, no un miembro")
    void invitarMiembro_solicitanteMiembro_lanzaForbiddenException() {
        // Dado: el solicitante es un miembro, no el owner
        InviteMemberRequest peticion = crearPeticionInvitacion("ajeno");
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));

        // Cuando + Entonces: se lanza ForbiddenException
        assertThrows(ForbiddenException.class,
                () -> projectService.inviteMember(10L, peticion, miembro));
    }

    // =========================================================
    // TU02-26 a TU02-29 — Remover miembro
    // =========================================================

    @Test
    @DisplayName("TU02-26: El owner puede remover a un miembro del proyecto")
    void removerMiembro_porOwner_eliminaMembresia() {
        // Dado: el miembro existe en el proyecto
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(userRepository.findById(2L)).thenReturn(Optional.of(miembro));
        when(projectMemberRepository.findByProjectAndUser(proyecto, miembro))
                .thenReturn(Optional.of(membresia));

        // Cuando: el owner remueve al miembro
        projectService.removeMember(10L, 2L, owner);

        // Entonces: la membresía se elimina
        verify(projectMemberRepository, times(1)).delete(membresia);
    }

    @Test
    @DisplayName("TU02-27: Al remover un miembro, sus tareas asignadas quedan sin responsable")
    void removerMiembro_desasignaTareasDelMiembro() {
        // Dado: el miembro tiene tareas asignadas en el proyecto
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(userRepository.findById(2L)).thenReturn(Optional.of(miembro));
        when(projectMemberRepository.findByProjectAndUser(proyecto, miembro))
                .thenReturn(Optional.of(membresia));

        // Cuando: se remueve al miembro
        projectService.removeMember(10L, 2L, owner);

        // Entonces: se llama al método que desasigna las tareas (assignedTo = null)
        verify(taskRepository, times(1)).unassignTasksFromUserInProject(proyecto, miembro);
    }

    @Test
    @DisplayName("TU02-28: El owner no puede ser removido del proyecto")
    void removerMiembro_intentaRemoverOwner_lanzaForbiddenException() {
        // Dado: el userId a remover es el del owner
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));

        // Cuando + Entonces: se lanza ForbiddenException
        assertThrows(ForbiddenException.class,
                () -> projectService.removeMember(10L, 1L, owner));
    }

    @Test
    @DisplayName("TU02-29: Un miembro NO puede remover a otros miembros")
    void removerMiembro_solicitanteMiembro_lanzaForbiddenException() {
        // Dado: el solicitante es un miembro, no el owner
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));

        // Cuando + Entonces: se lanza ForbiddenException
        assertThrows(ForbiddenException.class,
                () -> projectService.removeMember(10L, 3L, miembro));
    }

    // =========================================================
    // TU02-30 a TU02-33 — Listar miembros
    // =========================================================

    @Test
    @DisplayName("TU02-30: Listar miembros incluye al owner marcado como tal")
    void listarMiembros_incluyeOwnerConMarcaDeOwner() {
        // Dado: el proyecto tiene al owner y 2 miembros
        User miembro2 = buildUser(4L, "miembro2", "m2@mail.com");
        ProjectMember membresia2 = ProjectMember.builder()
                .id(2L).project(proyecto).user(miembro2).joinedAt(LocalDateTime.now()).build();
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, owner)).thenReturn(false);
        when(projectMemberRepository.findAllByProject(proyecto))
                .thenReturn(List.of(membresia, membresia2));

        // Cuando: se lista
        List<MemberResponse> lista = projectService.listMembers(10L, owner);

        // Entonces: hay 3 elementos y el primero (el owner) tiene isOwner=true
        assertEquals(3, lista.size());
        assertTrue(lista.get(0).isOwner());
    }

    @Test
    @DisplayName("TU02-31: El DTO de un miembro incluye la fecha en que se unió (joinedAt)")
    void listarMiembros_miembroTieneJoinedAt() {
        // Dado: el proyecto tiene 1 miembro con joinedAt
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, owner)).thenReturn(false);
        when(projectMemberRepository.findAllByProject(proyecto)).thenReturn(List.of(membresia));

        // Cuando: se lista
        List<MemberResponse> lista = projectService.listMembers(10L, owner);

        // Entonces: el segundo elemento (el miembro) tiene joinedAt no nulo
        MemberResponse miembroDTO = lista.get(1);
        assertNotNull(miembroDTO.getJoinedAt());
        assertFalse(miembroDTO.isOwner());
    }

    @Test
    @DisplayName("TU02-32: Un miembro puede listar los miembros del proyecto")
    void listarMiembros_porMiembro_retornaLista() {
        // Dado: el solicitante es un miembro activo
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, miembro)).thenReturn(true);
        when(projectMemberRepository.findAllByProject(proyecto)).thenReturn(List.of(membresia));

        // Cuando + Entonces: el miembro puede listar sin error
        assertDoesNotThrow(() -> projectService.listMembers(10L, miembro));
    }

    @Test
    @DisplayName("TU02-33: Un usuario ajeno NO puede listar los miembros del proyecto")
    void listarMiembros_porUsuarioAjeno_lanzaForbiddenException() {
        // Dado: el solicitante no es owner ni miembro
        when(projectRepository.findById(10L)).thenReturn(Optional.of(proyecto));
        when(projectMemberRepository.existsByProjectAndUser(proyecto, usuarioAjeno)).thenReturn(false);

        // Cuando + Entonces: se lanza ForbiddenException
        assertThrows(ForbiddenException.class,
                () -> projectService.listMembers(10L, usuarioAjeno));
    }

    // =========================================================
    // Métodos de ayuda para construir objetos de prueba
    // =========================================================

    private User buildUser(Long id, String username, String email) {
        return User.builder()
                .id(id)
                .username(username)
                .email(email)
                .password("hash")
                .createdAt(LocalDateTime.now())
                .build();
    }

    private Task buildTask(Long id, TaskStatus status) {
        return Task.builder()
                .id(id)
                .title("Tarea " + id)
                .description("Descripción")
                .status(status)
                .project(proyecto)
                .createdBy(owner)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
    }

    private CreateProjectRequest crearPeticionProyecto(String nombre, String descripcion) {
        CreateProjectRequest r = new CreateProjectRequest();
        r.setName(nombre);
        r.setDescription(descripcion);
        return r;
    }

    private UpdateProjectRequest crearPeticionActualizacion(String nombre, String descripcion) {
        UpdateProjectRequest r = new UpdateProjectRequest();
        r.setName(nombre);
        r.setDescription(descripcion);
        return r;
    }

    private InviteMemberRequest crearPeticionInvitacion(String username) {
        InviteMemberRequest r = new InviteMemberRequest();
        r.setUsername(username);
        return r;
    }
}