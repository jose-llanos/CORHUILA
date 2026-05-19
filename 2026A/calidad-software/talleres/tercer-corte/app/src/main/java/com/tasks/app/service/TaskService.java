package com.tasks.app.service;

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
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class TaskService {

    private final TaskRepository taskRepository;
    private final ProjectRepository projectRepository;
    private final ProjectMemberRepository projectMemberRepository;
    private final UserRepository userRepository;

    @Transactional
    public TaskResponse createTask(Long projectId, CreateTaskRequest request, User currentUser) {
        Project project = findProject(projectId);
        validateAccess(project, currentUser);
        Task task = Task.builder()
                .title(request.getTitle())
                .description(request.getDescription())
                .status(TaskStatus.PENDING)
                .project(project)
                .createdBy(currentUser)
                .build();
        return TaskResponse.from(taskRepository.save(task));
    }

    @Transactional
    public TaskResponse updateTask(Long projectId, Long taskId, UpdateTaskRequest request, User currentUser) {
        Project project = findProject(projectId);
        validateAccess(project, currentUser);
        Task task = findTaskInProject(taskId, project);
        task.setTitle(request.getTitle());
        task.setDescription(request.getDescription());
        return TaskResponse.from(taskRepository.save(task));
    }

    @Transactional
    public void deleteTask(Long projectId, Long taskId, User currentUser) {
        Project project = findProject(projectId);
        validateOwner(project, currentUser);
        Task task = findTaskInProject(taskId, project);
        taskRepository.delete(task);
    }

    @Transactional
    public TaskResponse changeStatus(Long projectId, Long taskId, ChangeTaskStatusRequest request, User currentUser) {
        Project project = findProject(projectId);
        Task task = findTaskInProject(taskId, project);
        boolean isOwner = project.getOwner().getId().equals(currentUser.getId());
        if (isOwner) {
            task.setStatus(request.getStatus());
        } else {
            if (!projectMemberRepository.existsByProjectAndUser(project, currentUser)) {
                throw new ForbiddenException("No tienes acceso a este proyecto");
            }
            // RF-03.4: miembro solo puede cambiar estado de sus tareas asignadas
            if (task.getAssignedTo() == null || !task.getAssignedTo().getId().equals(currentUser.getId())) {
                throw new ForbiddenException("Solo puedes cambiar el estado de las tareas asignadas a ti");
            }
            task.setStatus(request.getStatus());
        }
        return TaskResponse.from(taskRepository.save(task));
    }

    @Transactional
    public TaskResponse assignTask(Long projectId, Long taskId, AssignTaskRequest request, User currentUser) {
        Project project = findProject(projectId);
        validateOwner(project, currentUser);
        Task task = findTaskInProject(taskId, project);
        if (request.getAssignedToUserId() == null) {
            task.setAssignedTo(null);
        } else {
            User assignee = userRepository.findById(request.getAssignedToUserId())
                    .orElseThrow(() -> new ResourceNotFoundException("Usuario no encontrado"));
            boolean isOwner = assignee.getId().equals(project.getOwner().getId());
            boolean isMember = projectMemberRepository.existsByProjectAndUser(project, assignee);
            if (!isOwner && !isMember) {
                throw new ForbiddenException("El usuario no pertenece al proyecto");
            }
            task.setAssignedTo(assignee);
        }
        return TaskResponse.from(taskRepository.save(task));
    }

    @Transactional(readOnly = true)
    public List<TaskResponse> listTasks(Long projectId, User currentUser) {
        Project project = findProject(projectId);
        validateAccess(project, currentUser);
        return taskRepository.findAllByProject(project).stream()
                .map(TaskResponse::from)
                .toList();
    }

    private Project findProject(Long projectId) {
        return projectRepository.findById(projectId)
                .orElseThrow(() -> new ResourceNotFoundException("Proyecto no encontrado"));
    }

    private Task findTaskInProject(Long taskId, Project project) {
        Task task = taskRepository.findById(taskId)
                .orElseThrow(() -> new ResourceNotFoundException("Tarea no encontrada"));
        if (!task.getProject().getId().equals(project.getId())) {
            throw new ResourceNotFoundException("La tarea no pertenece al proyecto");
        }
        return task;
    }

    private void validateOwner(Project project, User user) {
        if (!project.getOwner().getId().equals(user.getId())) {
            throw new ForbiddenException("Solo el owner puede realizar esta acción");
        }
    }

    private void validateAccess(Project project, User user) {
        boolean isOwner = project.getOwner().getId().equals(user.getId());
        boolean isMember = projectMemberRepository.existsByProjectAndUser(project, user);
        if (!isOwner && !isMember) {
            throw new ForbiddenException("No tienes acceso a este proyecto");
        }
    }
}
