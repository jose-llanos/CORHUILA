package com.tasks.app.dto.response;

import com.tasks.app.entity.Task;
import com.tasks.app.entity.TaskStatus;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;

@Getter
@AllArgsConstructor
@Builder
public class TaskResponse {

    private Long id;
    private String title;
    private String description;
    private TaskStatus status;
    private UserProfileResponse assignedTo;
    private UserProfileResponse createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public static TaskResponse from(Task task) {
        return TaskResponse.builder()
                .id(task.getId())
                .title(task.getTitle())
                .description(task.getDescription())
                .status(task.getStatus())
                .assignedTo(task.getAssignedTo() != null
                        ? UserProfileResponse.from(task.getAssignedTo()) : null)
                .createdBy(UserProfileResponse.from(task.getCreatedBy()))
                .createdAt(task.getCreatedAt())
                .updatedAt(task.getUpdatedAt())
                .build();
    }
}
