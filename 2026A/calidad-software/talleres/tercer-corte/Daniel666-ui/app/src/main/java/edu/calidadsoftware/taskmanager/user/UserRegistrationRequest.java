package edu.calidadsoftware.taskmanager.user;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import javax.validation.constraints.Email;
import javax.validation.constraints.NotBlank;
import javax.validation.constraints.Size;

/**
 * DTO para registro de usuarios vía API (/api/users).
 *
 * Se valida con Bean Validation antes de pasar a la capa de servicio.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UserRegistrationRequest {

    @NotBlank(message = "Username is required")
    @Size(min = 3, max = 50, message = "Username must have between 3 and 50 characters")
    private String username;

    @NotBlank(message = "Email is required")
    @Email(message = "Email must be valid")
    @Size(max = 120, message = "Email must have at most 120 characters")
    private String email;

    @NotBlank(message = "Password is required")
    @Size(min = 4, max = 120, message = "Password must have between 4 and 120 characters")
    private String password;
}
