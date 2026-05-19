package com.tasks.app.unit.service;

import com.tasks.app.dto.request.LoginRequest;
import com.tasks.app.dto.request.RegisterRequest;
import com.tasks.app.dto.response.AuthResponse;
import com.tasks.app.dto.response.UserProfileResponse;
import com.tasks.app.entity.User;
import com.tasks.app.exception.ConflictException;
import com.tasks.app.exception.UnauthorizedException;
import com.tasks.app.repository.UserRepository;
import com.tasks.app.security.JwtService;
import com.tasks.app.service.UserService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.time.LocalDateTime;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

/*
 * Pruebas unitarias de UserService.
 *
 * Cada prueba sigue el patrón Dado / Cuando / Entonces:
 *   - Dado:   configuramos los "dobles" (mocks) para simular el comportamiento de la BD.
 *   - Cuando: llamamos al método del servicio que queremos probar.
 *   - Entonces: verificamos que el resultado es el esperado.
 *
 * Los repositorios y dependencias externas son SIMULADOS con Mockito
 * (no se toca ninguna base de datos real).
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("TU-01 — UserService: Gestión de Usuarios")
public class UserServiceTest {

    // "Dobles" (mocks) de las dependencias del servicio
    @Mock
    private UserRepository userRepository;

    @Mock
    private PasswordEncoder passwordEncoder;

    @Mock
    private JwtService jwtService;

    // El servicio real, pero con sus dependencias reemplazadas por los mocks
    @InjectMocks
    private UserService userService;

    // Usuario de ejemplo reutilizable en varios tests
    private User usuarioEjemplo;

    @BeforeEach
    void prepararDatos() {
        usuarioEjemplo = User.builder()
                .id(1L)
                .username("juan")
                .email("juan@mail.com")
                .password("$2a$10$hashBCrypt")
                .createdAt(LocalDateTime.of(2026, 1, 15, 10, 0))
                .build();
    }

    // =========================================================
    // TU01-01 y TU01-08 — Registro exitoso
    // =========================================================

    @Test
    @DisplayName("TU01-01: Registro exitoso — guarda el usuario y retorna sus datos")
    void registro_datosValidos_guardaYRetornaDatos() {
        // Dado: username y email no existen en BD, el encoder produce un hash
        RegisterRequest peticion = crearPeticionRegistro("juan", "juan@mail.com", "Pass1234");
        when(userRepository.existsByUsername("juan")).thenReturn(false);
        when(userRepository.existsByEmail("juan@mail.com")).thenReturn(false);
        when(passwordEncoder.encode("Pass1234")).thenReturn("$2a$10$hashBCrypt");
        when(userRepository.save(any(User.class))).thenReturn(usuarioEjemplo);

        // Cuando: se registra el usuario
        UserProfileResponse respuesta = userService.register(peticion);

        // Entonces: la respuesta contiene los datos correctos
        assertEquals("juan", respuesta.getUsername());
        assertEquals("juan@mail.com", respuesta.getEmail());
        assertNotNull(respuesta.getId());
        // Y se llamó a save exactamente una vez
        verify(userRepository, times(1)).save(any(User.class));
    }

    @Test
    @DisplayName("TU01-08: La contraseña se guarda hasheada, no en texto plano")
    void registro_passwordGuardadaHasheada() {
        // Dado: el registro es válido
        RegisterRequest peticion = crearPeticionRegistro("juan", "juan@mail.com", "Pass1234");
        when(userRepository.existsByUsername(anyString())).thenReturn(false);
        when(userRepository.existsByEmail(anyString())).thenReturn(false);
        when(passwordEncoder.encode("Pass1234")).thenReturn("$2a$10$hashBCrypt");
        when(userRepository.save(any(User.class))).thenReturn(usuarioEjemplo);

        // Cuando: se registra el usuario
        userService.register(peticion);

        // Entonces: el encoder fue llamado (la contraseña nunca se guarda en texto plano)
        verify(passwordEncoder, times(1)).encode("Pass1234");
    }

    // =========================================================
    // TU01-02 y TU01-03 — Registro falla por duplicados
    // =========================================================

    @Test
    @DisplayName("TU01-02: Registro falla si el username ya está en uso")
    void registro_usernameRepetido_lanzaConflictException() {
        // Dado: el username ya existe en BD
        RegisterRequest peticion = crearPeticionRegistro("juan", "otro@mail.com", "Pass1234");
        when(userRepository.existsByUsername("juan")).thenReturn(true);

        // Cuando + Entonces: el servicio lanza ConflictException
        assertThrows(ConflictException.class, () -> userService.register(peticion));

        // Y el usuario NUNCA se guarda
        verify(userRepository, never()).save(any());
    }

    @Test
    @DisplayName("TU01-03: Registro falla si el email ya está en uso")
    void registro_emailRepetido_lanzaConflictException() {
        // Dado: el email ya existe en BD
        RegisterRequest peticion = crearPeticionRegistro("nuevousuario", "juan@mail.com", "Pass1234");
        when(userRepository.existsByUsername("nuevousuario")).thenReturn(false);
        when(userRepository.existsByEmail("juan@mail.com")).thenReturn(true);

        // Cuando + Entonces: el servicio lanza ConflictException
        assertThrows(ConflictException.class, () -> userService.register(peticion));

        // Y el usuario NUNCA se guarda
        verify(userRepository, never()).save(any());
    }

    // =========================================================
    // TU01-09, TU01-10, TU01-11 — Login
    // =========================================================

    @Test
    @DisplayName("TU01-09: Login exitoso — retorna un token JWT no vacío")
    void login_credencialesValidas_retornaToken() {
        // Dado: el usuario existe y la contraseña coincide
        LoginRequest peticion = crearPeticionLogin("juan", "Pass1234");
        when(userRepository.findByUsername("juan")).thenReturn(Optional.of(usuarioEjemplo));
        when(passwordEncoder.matches("Pass1234", "$2a$10$hashBCrypt")).thenReturn(true);
        when(jwtService.generateToken(usuarioEjemplo)).thenReturn("token.jwt.valido");

        // Cuando: se hace login
        AuthResponse respuesta = userService.login(peticion);

        // Entonces: el token está presente y no está vacío
        assertNotNull(respuesta.getToken());
        assertFalse(respuesta.getToken().isEmpty());
    }

    @Test
    @DisplayName("TU01-10: Login falla con contraseña incorrecta")
    void login_passwordIncorrecta_lanzaUnauthorizedException() {
        // Dado: el usuario existe pero la contraseña no coincide
        LoginRequest peticion = crearPeticionLogin("juan", "ClaveErrada");
        when(userRepository.findByUsername("juan")).thenReturn(Optional.of(usuarioEjemplo));
        when(passwordEncoder.matches("ClaveErrada", "$2a$10$hashBCrypt")).thenReturn(false);

        // Cuando + Entonces: se lanza UnauthorizedException
        assertThrows(UnauthorizedException.class, () -> userService.login(peticion));
    }

    @Test
    @DisplayName("TU01-11: Login falla si el username no existe")
    void login_usuarioNoExiste_lanzaUnauthorizedException() {
        // Dado: el username no está registrado en BD
        LoginRequest peticion = crearPeticionLogin("usuariofantasma", "Pass1234");
        when(userRepository.findByUsername("usuariofantasma")).thenReturn(Optional.empty());

        // Cuando + Entonces: se lanza UnauthorizedException
        assertThrows(UnauthorizedException.class, () -> userService.login(peticion));
    }

    // =========================================================
    // TU01-13 — Ver perfil propio
    // =========================================================

    @Test
    @DisplayName("TU01-13: getProfile retorna id, username, email y createdAt del usuario autenticado")
    void obtenerPerfil_usuarioAutenticado_retornaDatosCompletos() {
        // Dado: un usuario autenticado (ya viene resuelto por el filtro JWT, no se consulta BD)

        // Cuando: se consulta su perfil
        UserProfileResponse perfil = userService.getProfile(usuarioEjemplo);

        // Entonces: el DTO incluye todos los campos visibles (sin password)
        assertEquals(1L, perfil.getId());
        assertEquals("juan", perfil.getUsername());
        assertEquals("juan@mail.com", perfil.getEmail());
        assertNotNull(perfil.getCreatedAt());
    }

    // =========================================================
    // Métodos de ayuda para construir objetos de prueba
    // =========================================================

    private RegisterRequest crearPeticionRegistro(String username, String email, String password) {
        RegisterRequest r = new RegisterRequest();
        r.setUsername(username);
        r.setEmail(email);
        r.setPassword(password);
        return r;
    }

    private LoginRequest crearPeticionLogin(String username, String password) {
        LoginRequest r = new LoginRequest();
        r.setUsername(username);
        r.setPassword(password);
        return r;
    }
}