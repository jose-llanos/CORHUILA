package com.tasks.app.dto.response;

import com.tasks.app.entity.Project;
import com.tasks.app.entity.Task;
import com.tasks.app.entity.TaskStatus;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

import java.util.List;

@Getter
@AllArgsConstructor
@Builder
public class ProjectDetailResponse {

    private Long id;
    private String name;
    private String description;
    private UserProfileResponse owner;
    private List<TaskResponse> pending;
    private List<TaskResponse> inProgress;
    private List<TaskResponse> done;

    public static ProjectDetailResponse from(Project project, List<Task> tasks) {
        return ProjectDetailResponse.builder()
                .id(project.getId())
                .name(project.getName())
                .description(project.getDescription())
                .owner(UserProfileResponse.from(project.getOwner()))
                .pending(tasks.stream()
                        .filter(t -> t.getStatus() == TaskStatus.PENDING)
                        .map(TaskResponse::from)
                        .toList())
                .inProgress(tasks.stream()
                        .filter(t -> t.getStatus() == TaskStatus.IN_PROGRESS)
                        .map(TaskResponse::from)
                        .toList())
                .done(tasks.stream()
                        .filter(t -> t.getStatus() == TaskStatus.DONE)
                        .map(TaskResponse::from)
                        .toList())
                .build();
    }
}
