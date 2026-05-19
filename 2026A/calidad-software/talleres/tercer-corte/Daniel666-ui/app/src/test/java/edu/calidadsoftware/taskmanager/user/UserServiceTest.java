package edu.calidadsoftware.taskmanager.user;

import edu.calidadsoftware.taskmanager.common.DuplicateResourceException;
import edu.calidadsoftware.taskmanager.common.ResourceNotFoundException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.security.crypto.password.PasswordEncoder;

import javax.validation.Validation;
import javax.validation.Validator;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Pruebas unitarias de UserService con Mockito.
 *
 * Incluye escenarios de registro, duplicados, validación de email y login fallido.
 */
@DisplayName("UserService")
class UserServiceTest {

    private UserRepository userRepository;
    private PasswordEncoder passwordEncoder;
    private Validator validator;
    private UserService userService;

    @BeforeEach
    void setUp() {
        userRepository = Mockito.mock(UserRepository.class);
        passwordEncoder = Mockito.mock(PasswordEncoder.class);
        validator = Validation.buildDefaultValidatorFactory().getValidator();
        userService = new UserService(userRepository, passwordEncoder, validator);
    }

    @AfterEach
    void tearDown() {
        Mockito.reset(userRepository, passwordEncoder);
    }

    private static UserRegistrationRequest validRequest() {
        return UserRegistrationRequest.builder()
                .username("newuser")
                .email("newuser@example.com")
                .password("pass1234")
                .build();
    }

    @Nested
    @DisplayName("register")
    class Register {

        @Test
        @DisplayName("Registra un usuario válido (role USER) y codifica password")
        void register_success() {
            UserRegistrationRequest request = validRequest();

            when(userRepository.existsByUsername("newuser")).thenReturn(false);
            when(userRepository.existsByEmail("newuser@example.com")).thenReturn(false);
            when(passwordEncoder.encode("pass1234")).thenReturn("{bcrypt}ENC");
            when(userRepository.save(any(User.class))).thenAnswer(inv -> {
                User u = inv.getArgument(0);
                u.setId(1L);
                return u;
            });

            User created = userService.register(request);

            assertNotNull(created);
            assertEquals(1L, created.getId());
            assertEquals("newuser", created.getUsername());
            assertEquals("newuser@example.com", created.getEmail());
            assertEquals(UserRole.USER, created.getRole());
            assertEquals("{bcrypt}ENC", created.getPassword());

            verify(userRepository, times(1)).existsByUsername("newuser");
            verify(userRepository, times(1)).existsByEmail("newuser@example.com");
            verify(passwordEncoder, times(1)).encode("pass1234");
            verify(userRepository, times(1)).save(any(User.class));
        }

        @Test
        @DisplayName("Falla si el username ya existe")
        void register_duplicateUsername_throws() {
            UserRegistrationRequest request = validRequest();
            when(userRepository.existsByUsername("newuser")).thenReturn(true);

            assertThrows(DuplicateResourceException.class, () -> userService.register(request));

            verify(userRepository, times(1)).existsByUsername("newuser");
            verify(userRepository, never()).existsByEmail(any());
            verify(userRepository, never()).save(any(User.class));
        }

        @Test
        @DisplayName("Falla si el email ya existe")
        void register_duplicateEmail_throws() {
            UserRegistrationRequest request = validRequest();
            when(userRepository.existsByUsername("newuser")).thenReturn(false);
            when(userRepository.existsByEmail("newuser@example.com")).thenReturn(true);

            assertThrows(DuplicateResourceException.class, () -> userService.register(request));

            verify(userRepository, times(1)).existsByUsername("newuser");
            verify(userRepository, times(1)).existsByEmail("newuser@example.com");
            verify(userRepository, never()).save(any(User.class));
        }

        @Test
        @DisplayName("Falla si el email es inválido")
        void register_invalidEmail_throws() {
            UserRegistrationRequest request = validRequest();
            request.setEmail("invalid-email");

            assertThrows(IllegalArgumentException.class, () -> userService.register(request));
            verify(userRepository, never()).save(any(User.class));
        }

        @Test
        @DisplayName("Falla si el username es vacío")
        void register_blankUsername_throws() {
            UserRegistrationRequest request = validRequest();
            request.setUsername(" ");

            assertThrows(IllegalArgumentException.class, () -> userService.register(request));
            verify(userRepository, never()).save(any(User.class));
        }
    }

    @Nested
    @DisplayName("login")
    class Login {

        @Test
        @DisplayName("Login falla si el usuario no existe")
        void login_userNotFound() {
            when(userRepository.findByUsername("missing")).thenReturn(Optional.empty());

            assertThrows(ResourceNotFoundException.class, () -> userService.login("missing", "admin"));
            verify(userRepository, times(1)).findByUsername("missing");
        }

        @Test
        @DisplayName("Login falla si el password no coincide")
        void login_failedPassword() {
            User existing = User.builder()
                    .id(10L)
                    .username("user")
                    .email("user@example.com")
                    .password("{noop}user")
                    .role(UserRole.USER)
                    .build();

            when(userRepository.findByUsername("user")).thenReturn(Optional.of(existing));
            when(passwordEncoder.matches(eq("wrong"), eq("{noop}user"))).thenReturn(false);

            assertThrows(IllegalArgumentException.class, () -> userService.login("user", "wrong"));
            verify(userRepository, times(1)).findByUsername("user");
            verify(passwordEncoder, times(1)).matches("wrong", "{noop}user");
        }

        @Test
        @DisplayName("Login exitoso devuelve el usuario")
        void login_success() {
            User existing = User.builder()
                    .id(11L)
                    .username("admin")
                    .email("admin@example.com")
                    .password("{noop}admin")
                    .role(UserRole.ADMIN)
                    .build();

            when(userRepository.findByUsername("admin")).thenReturn(Optional.of(existing));
            when(passwordEncoder.matches(eq("admin"), eq("{noop}admin"))).thenReturn(true);

            User logged = userService.login("admin", "admin");
            assertEquals(11L, logged.getId());
            assertEquals(UserRole.ADMIN, logged.getRole());
        }
    }

    @Nested
    @DisplayName("findById / delete")
    class FindAndDelete {

        @Test
        @DisplayName("findById falla si el id no existe")
        void findById_notFound() {
            when(userRepository.findById(99L)).thenReturn(Optional.empty());
            assertThrows(ResourceNotFoundException.class, () -> userService.findById(99L));
        }

        @Test
        @DisplayName("delete elimina si existe")
        void delete_success() {
            User existing = User.builder()
                    .id(5L)
                    .username("x")
                    .email("x@example.com")
                    .password("{noop}x")
                    .role(UserRole.USER)
                    .build();

            when(userRepository.findById(5L)).thenReturn(Optional.of(existing));

            userService.delete(5L);

            verify(userRepository, times(1)).findById(5L);
            verify(userRepository, times(1)).delete(eq(existing));
        }

        @Test
        @DisplayName("delete falla si el id no existe")
        void delete_notFound() {
            when(userRepository.findById(123L)).thenReturn(Optional.empty());

            assertThrows(ResourceNotFoundException.class, () -> userService.delete(123L));
            verify(userRepository, times(1)).findById(123L);
            verify(userRepository, never()).delete(any(User.class));
        }
    }
}
