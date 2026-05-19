package com.tasks.app.dto.request;

import lombok.Data;

@Data
public class AssignTaskRequest {

    // null = desasignar (RF-03.5)
    private Long assignedToUserId;
}
