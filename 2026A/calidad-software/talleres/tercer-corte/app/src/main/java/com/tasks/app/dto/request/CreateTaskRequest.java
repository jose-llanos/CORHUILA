package com.tasks.app.dto.request;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class CreateTaskRequest {

    @NotBlank
    @Size(min = 3, max = 150)
    private String title;

    @Size(max = 1000)
    private String description;
}
