package com.sgplab.backend.unitarias.service;

import com.sgplab.backend.dto.request.LoginRequest;
import com.sgplab.backend.dto.response.LoginResponse;
import com.sgplab.backend.exception.InvalidCredentialsException;
import com.sgplab.backend.model.entity.Usuario;
import com.sgplab.backend.model.enums.EstadoUsuario;
import com.sgplab.backend.model.enums.Rol;
import com.sgplab.backend.repository.IUsuarioRepository;
import com.sgplab.backend.security.JwtService;
import com.sgplab.backend.service.impl.AuthServiceImpl;
import com.sgplab.backend.util.PasswordHashUtil;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Pruebas unitarias para {@link AuthServiceImpl}.
 */
@ExtendWith(MockitoExtension.class)
class AuthServiceImplTest {

    @Mock
    private IUsuarioRepository usuarioRepository;

    @Mock
    private JwtService jwtService;

    @InjectMocks
    private AuthServiceImpl authService;

    private Usuario usuarioActivo;

    @BeforeEach
    void setUp() {
        usuarioActivo = new Usuario();
        usuarioActivo.setId(1L);
        usuarioActivo.setEmail("user@test.com");
        usuarioActivo.setNombre("Test");
        usuarioActivo.setRol(Rol.CLIENTE);
        usuarioActivo.setEstado(EstadoUsuario.ACTIVO);
        usuarioActivo.setPasswordHash(PasswordHashUtil.hash("correcta123"));
    }

    @Test
    @DisplayName("login: credenciales correctas devuelve LoginResponse con token")
    void login_Exito() {
        when(usuarioRepository.findByEmail("user@test.com")).thenReturn(Optional.of(usuarioActivo));
        when(jwtService.generateToken(usuarioActivo)).thenReturn("fake.jwt.token");
        when(jwtService.getExpirationMs()).thenReturn(3600000L);

        LoginResponse response = authService.login(new LoginRequest("user@test.com", "correcta123"));

        assertNotNull(response);
        assertEquals("fake.jwt.token", response.getToken());
        assertEquals("Bearer", response.getTokenType());
        assertEquals(1L, response.getUserId());
        assertEquals("user@test.com", response.getEmail());
        assertEquals(Rol.CLIENTE, response.getRol());
        verify(jwtService).generateToken(usuarioActivo);
    }

    @Test
    @DisplayName("login: email inexistente lanza InvalidCredentialsException")
    void login_EmailNoExiste() {
        when(usuarioRepository.findByEmail(any())).thenReturn(Optional.empty());

        assertThrows(InvalidCredentialsException.class,
                () -> authService.login(new LoginRequest("nadie@x.com", "x")));
        verify(jwtService, never()).generateToken(any());
    }

    @Test
    @DisplayName("login: password incorrecta lanza InvalidCredentialsException")
    void login_PasswordIncorrecta() {
        when(usuarioRepository.findByEmail("user@test.com")).thenReturn(Optional.of(usuarioActivo));

        assertThrows(InvalidCredentialsException.class,
                () -> authService.login(new LoginRequest("user@test.com", "incorrecta")));
        verify(jwtService, never()).generateToken(any());
    }

    @Test
    @DisplayName("login: usuario PENALIZADO no puede ingresar")
    void login_UsuarioPenalizado() {
        usuarioActivo.setEstado(EstadoUsuario.PENALIZADO);
        when(usuarioRepository.findByEmail("user@test.com")).thenReturn(Optional.of(usuarioActivo));

        InvalidCredentialsException ex = assertThrows(InvalidCredentialsException.class,
                () -> authService.login(new LoginRequest("user@test.com", "correcta123")));
        assertEquals("La cuenta no esta activa.", ex.getMessage());
        verify(jwtService, never()).generateToken(any());
    }
}
