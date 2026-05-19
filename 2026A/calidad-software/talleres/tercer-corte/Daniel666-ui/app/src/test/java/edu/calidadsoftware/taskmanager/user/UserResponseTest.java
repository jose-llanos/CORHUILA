package edu.calidadsoftware.taskmanager.user;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Pruebas unitarias para UserResponse.
 */
@DisplayName("UserResponse")
class UserResponseTest {

    @Test
    @DisplayName("from(User) mapea campos esperados")
    void from_mapsFields() {
        User user = User.builder()
                .id(1L)
                .username("u")
                .email("u@example.com")
                .password("{noop}x")
                .role(UserRole.ADMIN)
                .build();

        UserResponse dto = UserResponse.from(user);

        assertNotNull(dto);
        assertEquals(1L, dto.getId());
        assertEquals("u", dto.getUsername());
        assertEquals("u@example.com", dto.getEmail());
        assertEquals(UserRole.ADMIN, dto.getRole());
    }

    @Test
    @DisplayName("builder construye DTO correctamente")
    void builder_works() {
        UserResponse dto = UserResponse.builder()
                .id(2L)
                .username("x")
                .email("x@x.com")
                .role(UserRole.USER)
                .build();

        assertEquals(2L, dto.getId());
        assertEquals("x", dto.getUsername());
        assertEquals(UserRole.USER, dto.getRole());
    }
}

