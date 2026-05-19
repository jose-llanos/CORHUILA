package com.tasks.app.service;

import com.tasks.app.dto.request.CreateProjectRequest;
import com.tasks.app.dto.request.InviteMemberRequest;
import com.tasks.app.dto.request.UpdateProjectRequest;
import com.tasks.app.dto.response.MemberResponse;
import com.tasks.app.dto.response.ProjectDetailResponse;
import com.tasks.app.dto.response.ProjectResponse;
import com.tasks.app.entity.Project;
import com.tasks.app.entity.ProjectMember;
import com.tasks.app.entity.Task;
import com.tasks.app.entity.User;
import com.tasks.app.exception.ConflictException;
import com.tasks.app.exception.ForbiddenException;
import com.tasks.app.exception.ResourceNotFoundException;
import com.tasks.app.repository.ProjectMemberRepository;
import com.tasks.app.repository.ProjectRepository;
import com.tasks.app.repository.TaskRepository;
import com.tasks.app.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ProjectService {

    private final ProjectRepository projectRepository;
    private final ProjectMemberRepository projectMemberRepository;
    private final UserRepository userRepository;
    private final TaskRepository taskRepository;

    @Transactional
    public ProjectResponse createProject(CreateProjectRequest request, User currentUser) {
        if (projectRepository.existsByNameAndOwner(request.getName(), currentUser)) {
            throw new ConflictException("Ya tienes un proyecto con ese nombre");
        }
        Project project = Project.builder()
                .name(request.getName())
                .description(request.getDescription())
                .owner(currentUser)
                .build();
        return ProjectResponse.from(projectRepository.save(project));
    }

    @Transactional(readOnly = true)
    public List<ProjectResponse> listProjects(User currentUser) {
        return projectRepository.findAllAccessibleByUser(currentUser).stream()
                .map(ProjectResponse::from)
                .toList();
    }

    @Transactional(readOnly = true)
    public ProjectDetailResponse getProjectDetail(Long projectId, User currentUser) {
        Project project = findProject(projectId);
        validateAccess(project, currentUser);
        List<Task> tasks = taskRepository.findAllByProject(project);
        return ProjectDetailResponse.from(project, tasks);
    }

    @Transactional
    public ProjectResponse updateProject(Long projectId, UpdateProjectRequest request, User currentUser) {
        Project project = findProject(projectId);
        validateOwner(project, currentUser);
        if (!project.getName().equals(request.getName()) &&
                projectRepository.existsByNameAndOwner(request.getName(), currentUser)) {
            throw new ConflictException("Ya tienes un proyecto con ese nombre");
        }
        project.setName(request.getName());
        project.setDescription(request.getDescription());
        return ProjectResponse.from(projectRepository.save(project));
    }

    @Transactional
    public void deleteProject(Long projectId, User currentUser) {
        Project project = findProject(projectId);
        validateOwner(project, currentUser);
        projectRepository.delete(project);
    }

    @Transactional
    public MemberResponse inviteMember(Long projectId, InviteMemberRequest request, User currentUser) {
        Project project = findProject(projectId);
        validateOwner(project, currentUser);
        User invited = userRepository.findByUsername(request.getUsername())
                .orElseThrow(() -> new ResourceNotFoundException("Usuario no encontrado"));
        if (invited.getId().equals(project.getOwner().getId())) {
            throw new ConflictException("El owner no puede ser invitado como miembro");
        }
        if (projectMemberRepository.existsByProjectAndUser(project, invited)) {
            throw new ConflictException("El usuario ya es miembro del proyecto");
        }
        ProjectMember member = ProjectMember.builder()
                .project(project)
                .user(invited)
                .build();
        return MemberResponse.fromMember(projectMemberRepository.save(member));
    }

    @Transactional
    public void removeMember(Long projectId, Long userId, User currentUser) {
        Project project = findProject(projectId);
        validateOwner(project, currentUser);
        if (userId.equals(project.getOwner().getId())) {
            throw new ForbiddenException("No se puede remover al owner del proyecto");
        }
        User target = userRepository.findById(userId)
                .orElseThrow(() -> new ResourceNotFoundException("Usuario no encontrado"));
        ProjectMember membership = projectMemberRepository.findByProjectAndUser(project, target)
                .orElseThrow(() -> new ResourceNotFoundException("El usuario no es miembro del proyecto"));
        taskRepository.unassignTasksFromUserInProject(project, target);
        projectMemberRepository.delete(membership);
    }

    @Transactional(readOnly = true)
    public List<MemberResponse> listMembers(Long projectId, User currentUser) {
        Project project = findProject(projectId);
        validateAccess(project, currentUser);
        List<MemberResponse> members = new ArrayList<>();
        members.add(MemberResponse.fromOwner(project.getOwner()));
        projectMemberRepository.findAllByProject(project).stream()
                .map(MemberResponse::fromMember)
                .forEach(members::add);
        return members;
    }

    private Project findProject(Long projectId) {
        return projectRepository.findById(projectId)
                .orElseThrow(() -> new ResourceNotFoundException("Proyecto no encontrado"));
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
