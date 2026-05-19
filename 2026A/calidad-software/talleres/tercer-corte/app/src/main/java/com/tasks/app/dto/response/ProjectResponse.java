package com.tasks.app.dto.response;

import com.tasks.app.entity.Project;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@AllArgsConstructor
@Builder
public class ProjectResponse {

    private Long id;
    private String name;
    private String description;
    private UserProfileResponse owner;

    public static ProjectResponse from(Project project) {
        return ProjectResponse.builder()
                .id(project.getId())
                .name(project.getName())
                .description(project.getDescription())
                .owner(UserProfileResponse.from(project.getOwner()))
                .build();
    }
}
