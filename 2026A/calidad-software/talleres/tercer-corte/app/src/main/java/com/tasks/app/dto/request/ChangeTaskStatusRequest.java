package com.tasks.app.dto.request;

import com.tasks.app.entity.TaskStatus;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class ChangeTaskStatusRequest {

    @NotNull
    private TaskStatus status;
}
