package com.sgplab.backend.unitarias.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sgplab.backend.controller.UsuarioController;
import com.sgplab.backend.dto.request.UsuarioRequest;
import com.sgplab.backend.dto.response.UsuarioResponse;
import com.sgplab.backend.exception.DuplicateResourceException;
import com.sgplab.backend.exception.GlobalExceptionHandler;
import com.sgplab.backend.exception.ResourceNotFoundException;
import com.sgplab.backend.model.enums.EstadoUsuario;
import com.sgplab.backend.model.enums.Rol;
import com.sgplab.backend.service.contract.IUsuarioService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class UsuarioControllerTest {

    private MockMvc mockMvc;
    private IUsuarioService usuarioService;
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        usuarioService = Mockito.mock(IUsuarioService.class);
        UsuarioController controller = new UsuarioController(usuarioService);
        mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
        objectMapper = new ObjectMapper();
    }

    @Test
    @DisplayName("GET /api/usuarios -> 200 con lista")
    void obtenerTodos() throws Exception {
        Mockito.when(usuarioService.obtenerTodos()).thenReturn(List.of(
                new UsuarioResponse(1L, "Admin", "a@x.com", Rol.ADMINISTRADOR, EstadoUsuario.ACTIVO)
        ));
        mockMvc.perform(get("/api/usuarios"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].email").value("a@x.com"));
    }

    @Test
    @DisplayName("GET /api/usuarios/{id} OK")
    void obtenerPorId_OK() throws Exception {
        Mockito.when(usuarioService.obtenerPorId(1L))
                .thenReturn(new UsuarioResponse(1L, "Administrador", "x@x.com", Rol.CLIENTE, EstadoUsuario.ACTIVO));
        mockMvc.perform(get("/api/usuarios/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(1));
    }

    @Test
    @DisplayName("GET /api/usuarios/{id} -> 404 cuando no existe")
    void obtenerPorId_404() throws Exception {
        Mockito.when(usuarioService.obtenerPorId(99L))
                .thenThrow(new ResourceNotFoundException("Usuario", "id", 99L));
        mockMvc.perform(get("/api/usuarios/99"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.status").value(404));
    }

    @Test
    @DisplayName("POST /api/usuarios -> 201 cuando valido")
    void crear_OK() throws Exception {
        UsuarioRequest req = new UsuarioRequest();
        req.setNombre("Juan");
        req.setEmail("j@x.com");
        req.setPassword("password123");
        req.setRol(Rol.CLIENTE);

        Mockito.when(usuarioService.crear(any()))
                .thenReturn(new UsuarioResponse(7L, "Juan", "j@x.com", Rol.CLIENTE, EstadoUsuario.ACTIVO));

        mockMvc.perform(post("/api/usuarios")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").value(7));
    }

    @Test
    @DisplayName("POST /api/usuarios -> 400 cuando email invalido")
    void crear_400_EmailInvalido() throws Exception {
        UsuarioRequest req = new UsuarioRequest();
        req.setNombre("Carlos"); // Corregido de "X" a "Carlos" para que pase la validación de tamaño
        req.setEmail("no-es-email"); // Este forzará adecuadamente el 400 Bad Request por formato de email
        req.setPassword("password123");
        req.setRol(Rol.CLIENTE);

        mockMvc.perform(post("/api/usuarios")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.fieldErrors[?(@.field=='email')]").exists());
    }

    @Test
    @DisplayName("POST /api/usuarios -> 409 cuando duplicado")
    void crear_409() throws Exception {
        UsuarioRequest req = new UsuarioRequest();
        req.setNombre("Juan"); // Corregido de "J" a "Juan" para cumplir con @Size(min = 2)
        req.setEmail("j@x.com");
        req.setPassword("password123");
        req.setRol(Rol.CLIENTE);

        Mockito.when(usuarioService.crear(any()))
                .thenThrow(new DuplicateResourceException("Email duplicado"));

        mockMvc.perform(post("/api/usuarios")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isConflict());
    }

    @Test
    @DisplayName("PUT /api/usuarios/{id} OK")
    void actualizar_OK() throws Exception {
        UsuarioRequest req = new UsuarioRequest();
        req.setNombre("Juan");
        req.setEmail("juan@example.com");
        req.setPassword("password123");
        req.setRol(Rol.CLIENTE);

        Mockito.when(usuarioService.actualizar(eq(1L), any()))
                .thenReturn(new UsuarioResponse(1L, "Juan", "juan@example.com", Rol.CLIENTE, EstadoUsuario.ACTIVO));

        mockMvc.perform(put("/api/usuarios/1")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isOk());
    }

    @Test
    @DisplayName("DELETE /api/usuarios/{id} -> 204")
    void eliminar_OK() throws Exception {
        mockMvc.perform(delete("/api/usuarios/1"))
                .andExpect(status().isNoContent());
        Mockito.verify(usuarioService).eliminar(1L);
    }
}