package com.tasks.app.dto.response;

import com.tasks.app.entity.ProjectMember;
import com.tasks.app.entity.User;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;

@Getter
@AllArgsConstructor
@Builder
public class MemberResponse {

    private Long id;
    private String username;
    private String email;
    private LocalDateTime joinedAt;
    private boolean owner;

    public static MemberResponse fromOwner(User user) {
        return MemberResponse.builder()
                .id(user.getId())
                .username(user.getUsername())
                .email(user.getEmail())
                .joinedAt(null)
                .owner(true)
                .build();
    }

    public static MemberResponse fromMember(ProjectMember member) {
        return MemberResponse.builder()
                .id(member.getUser().getId())
                .username(member.getUser().getUsername())
                .email(member.getUser().getEmail())
                .joinedAt(member.getJoinedAt())
                .owner(false)
                .build();
    }
}
