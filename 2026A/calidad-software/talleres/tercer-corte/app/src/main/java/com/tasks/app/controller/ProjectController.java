package com.tasks.app.controller;

import com.tasks.app.dto.request.CreateProjectRequest;
import com.tasks.app.dto.request.InviteMemberRequest;
import com.tasks.app.dto.request.UpdateProjectRequest;
import com.tasks.app.dto.response.MemberResponse;
import com.tasks.app.dto.response.ProjectDetailResponse;
import com.tasks.app.dto.response.ProjectResponse;
import com.tasks.app.entity.User;
import com.tasks.app.service.ProjectService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/projects")
@RequiredArgsConstructor
public class ProjectController {

    private final ProjectService projectService;

    @PostMapping
    public ResponseEntity<ProjectResponse> create(
            @Valid @RequestBody CreateProjectRequest request,
            @AuthenticationPrincipal User currentUser) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(projectService.createProject(request, currentUser));
    }

    @GetMapping
    public ResponseEntity<List<ProjectResponse>> list(@AuthenticationPrincipal User currentUser) {
        return ResponseEntity.ok(projectService.listProjects(currentUser));
    }

    @GetMapping("/{id}")
    public ResponseEntity<ProjectDetailResponse> detail(
            @PathVariable Long id,
            @AuthenticationPrincipal User currentUser) {
        return ResponseEntity.ok(projectService.getProjectDetail(id, currentUser));
    }

    @PutMapping("/{id}")
    public ResponseEntity<ProjectResponse> update(
            @PathVariable Long id,
            @Valid @RequestBody UpdateProjectRequest request,
            @AuthenticationPrincipal User currentUser) {
        return ResponseEntity.ok(projectService.updateProject(id, request, currentUser));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(
            @PathVariable Long id,
            @AuthenticationPrincipal User currentUser) {
        projectService.deleteProject(id, currentUser);
        return ResponseEntity.noContent().build();
    }

    @PostMapping("/{id}/members")
    public ResponseEntity<MemberResponse> inviteMember(
            @PathVariable Long id,
            @Valid @RequestBody InviteMemberRequest request,
            @AuthenticationPrincipal User currentUser) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(projectService.inviteMember(id, request, currentUser));
    }

    @GetMapping("/{id}/members")
    public ResponseEntity<List<MemberResponse>> listMembers(
            @PathVariable Long id,
            @AuthenticationPrincipal User currentUser) {
        return ResponseEntity.ok(projectService.listMembers(id, currentUser));
    }

    @DeleteMapping("/{id}/members/{userId}")
    public ResponseEntity<Void> removeMember(
            @PathVariable Long id,
            @PathVariable Long userId,
            @AuthenticationPrincipal User currentUser) {
        projectService.removeMember(id, userId, currentUser);
        return ResponseEntity.noContent().build();
    }
}
