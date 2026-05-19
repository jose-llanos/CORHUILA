package com.tasks.app.dto.response;

import com.tasks.app.entity.User;
import lombok.Getter;
import lombok.AllArgsConstructor;
import lombok.Builder;

import java.time.LocalDateTime;

@Getter
@AllArgsConstructor
@Builder
public class UserProfileResponse {

    private Long id;
    private String username;
    private String email;
    private LocalDateTime createdAt;

    public static UserProfileResponse from(User user) {
        return UserProfileResponse.builder()
                .id(user.getId())
                .username(user.getUsername())
                .email(user.getEmail())
                .createdAt(user.getCreatedAt())
                .build();
    }
}
