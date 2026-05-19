package com.tasks.app.dto.request;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class UpdateProjectRequest {

    @NotBlank
    @Size(min = 3, max = 100)
    private String name;

    @Size(max = 500)
    private String description;
}
