package com.tasks.app.controller;

import com.tasks.app.dto.request.AssignTaskRequest;
import com.tasks.app.dto.request.ChangeTaskStatusRequest;
import com.tasks.app.dto.request.CreateTaskRequest;
import com.tasks.app.dto.request.UpdateTaskRequest;
import com.tasks.app.dto.response.TaskResponse;
import com.tasks.app.entity.User;
import com.tasks.app.service.TaskService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/projects/{projectId}/tasks")
@RequiredArgsConstructor
public class TaskController {

    private final TaskService taskService;

    @PostMapping
    public ResponseEntity<TaskResponse> create(
            @PathVariable Long projectId,
            @Valid @RequestBody CreateTaskRequest request,
            @AuthenticationPrincipal User currentUser) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(taskService.createTask(projectId, request, currentUser));
    }

    @GetMapping
    public ResponseEntity<List<TaskResponse>> list(
            @PathVariable Long projectId,
            @AuthenticationPrincipal User currentUser) {
        return ResponseEntity.ok(taskService.listTasks(projectId, currentUser));
    }

    @PutMapping("/{taskId}")
    public ResponseEntity<TaskResponse> update(
            @PathVariable Long projectId,
            @PathVariable Long taskId,
            @Valid @RequestBody UpdateTaskRequest request,
            @AuthenticationPrincipal User currentUser) {
        return ResponseEntity.ok(taskService.updateTask(projectId, taskId, request, currentUser));
    }

    @DeleteMapping("/{taskId}")
    public ResponseEntity<Void> delete(
            @PathVariable Long projectId,
            @PathVariable Long taskId,
            @AuthenticationPrincipal User currentUser) {
        taskService.deleteTask(projectId, taskId, currentUser);
        return ResponseEntity.noContent().build();
    }

    @PatchMapping("/{taskId}/status")
    public ResponseEntity<TaskResponse> changeStatus(
            @PathVariable Long projectId,
            @PathVariable Long taskId,
            @Valid @RequestBody ChangeTaskStatusRequest request,
            @AuthenticationPrincipal User currentUser) {
        return ResponseEntity.ok(taskService.changeStatus(projectId, taskId, request, currentUser));
    }

    @PatchMapping("/{taskId}/assign")
    public ResponseEntity<TaskResponse> assign(
            @PathVariable Long projectId,
            @PathVariable Long taskId,
            @RequestBody AssignTaskRequest request,
            @AuthenticationPrincipal User currentUser) {
        return ResponseEntity.ok(taskService.assignTask(projectId, taskId, request, currentUser));
    }
}
