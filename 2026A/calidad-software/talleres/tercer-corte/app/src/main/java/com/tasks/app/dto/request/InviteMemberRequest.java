package com.tasks.app.dto.request;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class InviteMemberRequest {

    @NotBlank
    private String username;
}
