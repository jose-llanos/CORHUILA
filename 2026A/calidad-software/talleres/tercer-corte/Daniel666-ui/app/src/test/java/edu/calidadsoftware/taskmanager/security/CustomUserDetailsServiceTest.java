package edu.calidadsoftware.taskmanager.security;

import edu.calidadsoftware.taskmanager.user.User;
import edu.calidadsoftware.taskmanager.user.UserRepository;
import edu.calidadsoftware.taskmanager.user.UserRole;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UsernameNotFoundException;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.when;

/**
 * Pruebas unitarias para CustomUserDetailsService.
 */
@DisplayName("CustomUserDetailsService")
class CustomUserDetailsServiceTest {

    @Test
    @DisplayName("Carga usuario existente y mapea ROLE_ADMIN")
    void loadUser_success() {
        UserRepository repo = Mockito.mock(UserRepository.class);
        CustomUserDetailsService service = new CustomUserDetailsService(repo);

        User user = User.builder()
                .id(1L)
                .username("admin")
                .email("admin@example.com")
                .password("{noop}admin")
                .role(UserRole.ADMIN)
                .build();

        when(repo.findByUsername("admin")).thenReturn(Optional.of(user));

        UserDetails details = service.loadUserByUsername("admin");
        assertEquals("admin", details.getUsername());
        assertEquals("{noop}admin", details.getPassword());
        assertEquals(1, details.getAuthorities().size());
        assertEquals("ROLE_ADMIN", details.getAuthorities().iterator().next().getAuthority());
    }

    @Test
    @DisplayName("Lanza UsernameNotFoundException si el usuario no existe")
    void loadUser_notFound() {
        UserRepository repo = Mockito.mock(UserRepository.class);
        CustomUserDetailsService service = new CustomUserDetailsService(repo);

        when(repo.findByUsername("missing")).thenReturn(Optional.empty());

        assertThrows(UsernameNotFoundException.class, () -> service.loadUserByUsername("missing"));
    }
}

