package edu.calidadsoftware.taskmanager.user;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

/**
 * DTO de salida para exponer usuarios sin información sensible (password).
 */
@Getter
@AllArgsConstructor
@Builder
public class UserResponse {

    private final Long id;
    private final String username;
    private final String email;
    private final UserRole role;

    public static UserResponse from(User user) {
        return UserResponse.builder()
                .id(user.getId())
                .username(user.getUsername())
                .email(user.getEmail())
                .role(user.getRole())
                .build();
    }
}
