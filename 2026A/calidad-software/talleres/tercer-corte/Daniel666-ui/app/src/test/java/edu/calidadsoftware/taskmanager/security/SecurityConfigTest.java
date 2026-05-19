package edu.calidadsoftware.taskmanager.security;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.security.crypto.password.PasswordEncoder;

import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Pruebas unitarias para SecurityConfig.
 *
 * Se invocan directamente los métodos configure para aumentar cobertura de configuración.
 */
@DisplayName("SecurityConfig")
class SecurityConfigTest {

    @Test
    @DisplayName("PasswordEncoder se crea correctamente")
    void passwordEncoder_created() {
        SecurityConfig config = new SecurityConfig(Mockito.mock(CustomUserDetailsService.class));
        PasswordEncoder encoder = config.passwordEncoder();
        assertNotNull(encoder);
    }
}
