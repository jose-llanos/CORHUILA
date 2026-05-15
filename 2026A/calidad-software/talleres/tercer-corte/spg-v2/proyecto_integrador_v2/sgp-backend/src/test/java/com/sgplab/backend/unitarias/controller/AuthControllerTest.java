package com.sgplab.backend.unitarias.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sgplab.backend.controller.AuthController;
import com.sgplab.backend.dto.request.LoginRequest;
import com.sgplab.backend.dto.response.LoginResponse;
import com.sgplab.backend.exception.GlobalExceptionHandler;
import com.sgplab.backend.exception.InvalidCredentialsException;
import com.sgplab.backend.model.enums.Rol;
import com.sgplab.backend.service.contract.IAuthService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import static org.mockito.ArgumentMatchers.any;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Tests del AuthController usando MockMvc standalone (sin contexto Spring completo).
 */
class AuthControllerTest {

    private MockMvc mockMvc;
    private IAuthService authService;
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        authService = Mockito.mock(IAuthService.class);
        AuthController controller = new AuthController(authService);
        mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
        objectMapper = new ObjectMapper();
    }

    @Test
    @DisplayName("POST /api/auth/login: credenciales correctas -> 200 con token")
    void login_OK() throws Exception {
        LoginResponse fake = new LoginResponse("token123", 3600000L, 1L, "u@x.com", "User", Rol.CLIENTE);
        Mockito.when(authService.login(any())).thenReturn(fake);

        LoginRequest req = new LoginRequest("u@x.com", "password");
        mockMvc.perform(post("/api/auth/login")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.token").value("token123"))
                .andExpect(jsonPath("$.tokenType").value("Bearer"))
                .andExpect(jsonPath("$.rol").value("CLIENTE"));
    }

    @Test
    @DisplayName("POST /api/auth/login: credenciales invalidas -> 401")
    void login_Unauthorized() throws Exception {
        Mockito.when(authService.login(any())).thenThrow(new InvalidCredentialsException("Credenciales invalidas."));
        LoginRequest req = new LoginRequest("u@x.com", "bad");

        mockMvc.perform(post("/api/auth/login")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.status").value(401))
                .andExpect(jsonPath("$.error").value("Unauthorized"));
    }

    @Test
    @DisplayName("POST /api/auth/login: payload invalido -> 400")
    void login_BadRequest() throws Exception {
        LoginRequest req = new LoginRequest("", "");
        mockMvc.perform(post("/api/auth/login")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.fieldErrors").exists());
    }
}
