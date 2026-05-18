package com.medicita.app.service.impl;

import com.medicita.app.dto.user.UserDTO;
import com.medicita.app.entity.User;
import com.medicita.app.enums.Role;
import com.medicita.app.exception.ResourceNotFoundException;
import com.medicita.app.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/*
 * Pruebas unitarias para UserServiceImpl.
 * La parte más interesante aquí es getCurrentUser(), que lee el email
 * del SecurityContext (el mecanismo de Spring Security para saber quién
 * está logueado). Para probarlo sin levantar todo el contexto de Spring,
 * mockeamos el SecurityContext manualmente.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("UserServiceImpl — Pruebas unitarias")
class UserServiceImplTest {

    @Mock private UserRepository userRepository;

    @InjectMocks
    private UserServiceImpl userService;

    private User user;

    @BeforeEach
    void setUp() {
        user = User.builder()
                .id(UUID.randomUUID())
                .firstName("María")
                .lastName("López")
                .email("maria@medicita.com")
                .role(Role.PATIENT)
                .active(true)
                .build();
    }

    // =========================================================================
    // findById()
    // =========================================================================

    @Test
    @DisplayName("findById: devuelve el UserDTO cuando el usuario existe")
    void findById_usuarioExiste_devuelveDTO() {
        when(userRepository.findById(user.getId())).thenReturn(Optional.of(user));

        UserDTO result = userService.findById(user.getId());

        assertThat(result.getEmail()).isEqualTo("maria@medicita.com");
        assertThat(result.getRole()).isEqualTo("PATIENT");
    }

    @Test
    @DisplayName("findById: lanza ResourceNotFoundException si el usuario no existe")
    void findById_noExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(userRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> userService.findById(fakeId))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    // =========================================================================
    // findAll()
    // =========================================================================

    @Test
    @DisplayName("findAll: devuelve todos los usuarios del sistema")
    void findAll_devuelveTodos() {
        when(userRepository.findAll()).thenReturn(List.of(user));

        List<UserDTO> result = userService.findAll();

        assertThat(result).hasSize(1);
        assertThat(result.get(0).getFirstName()).isEqualTo("María");
    }

    // =========================================================================
    // deactivate() y activate()
    // =========================================================================

    @Test
    @DisplayName("deactivate: pone active=false al usuario")
    void deactivate_usuarioExiste_loDesactiva() {
        when(userRepository.findById(user.getId())).thenReturn(Optional.of(user));

        userService.deactivate(user.getId());

        assertThat(user.isActive()).isFalse();
        verify(userRepository).save(user);
    }

    @Test
    @DisplayName("activate: pone active=true al usuario desactivado")
    void activate_usuarioInactivo_loActiva() {
        user.setActive(false);
        when(userRepository.findById(user.getId())).thenReturn(Optional.of(user));

        userService.activate(user.getId());

        assertThat(user.isActive()).isTrue();
        verify(userRepository).save(user);
    }

    @Test
    @DisplayName("deactivate: lanza ResourceNotFoundException si el usuario no existe")
    void deactivate_noExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(userRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> userService.deactivate(fakeId))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    @Test
    @DisplayName("activate: lanza ResourceNotFoundException si el usuario no existe")
    void activate_noExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(userRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> userService.activate(fakeId))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    // =========================================================================
    // getCurrentUser()
    // =========================================================================

    @Test
    @DisplayName("getCurrentUser: devuelve el usuario correspondiente al email del SecurityContext")
    void getCurrentUser_emailEnSecurityContext_devuelveUsuario() {
        // Simulamos el SecurityContext con un Authentication que tiene el email
        Authentication auth = mock(Authentication.class);
        when(auth.getName()).thenReturn("maria@medicita.com");

        SecurityContext securityContext = mock(SecurityContext.class);
        when(securityContext.getAuthentication()).thenReturn(auth);
        SecurityContextHolder.setContext(securityContext);

        when(userRepository.findByEmail("maria@medicita.com")).thenReturn(Optional.of(user));

        User result = userService.getCurrentUser();

        assertThat(result.getEmail()).isEqualTo("maria@medicita.com");

        // Limpiamos el contexto para no afectar otros tests
        SecurityContextHolder.clearContext();
    }

    @Test
    @DisplayName("getCurrentUser: lanza ResourceNotFoundException si el email no existe en BD")
    void getCurrentUser_emailNoEncontrado_lanzaExcepcion() {
        Authentication auth = mock(Authentication.class);
        when(auth.getName()).thenReturn("noexiste@medicita.com");

        SecurityContext securityContext = mock(SecurityContext.class);
        when(securityContext.getAuthentication()).thenReturn(auth);
        SecurityContextHolder.setContext(securityContext);

        when(userRepository.findByEmail("noexiste@medicita.com")).thenReturn(Optional.empty());

        assertThatThrownBy(() -> userService.getCurrentUser())
                .isInstanceOf(ResourceNotFoundException.class);

        SecurityContextHolder.clearContext();
    }
}
