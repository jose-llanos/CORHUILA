package com.sgplab.backend.integracion;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.sgplab.backend.dto.request.EquipoRequest;
import com.sgplab.backend.dto.request.LoginRequest;
import com.sgplab.backend.dto.request.PrestamoRequest;
import com.sgplab.backend.dto.request.UsuarioRequest;
import com.sgplab.backend.dto.response.LoginResponse;
import com.sgplab.backend.model.enums.Rol;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDate;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Tests de integracion end-to-end. Levantan el contexto completo de Spring,
 * H2 en memoria con datos de import-test.sql, y filtros JWT reales.
 *
 * <p>Flujos validados:
 * <ul>
 *   <li>Login con SHA-256 de admin y cliente preconfigurados.</li>
 *   <li>Acceso denegado sin token.</li>
 *   <li>Token invalido = 401.</li>
 *   <li>Cliente no puede tocar endpoints de admin (403).</li>
 *   <li>Flujo completo: login admin -> crea usuario cliente -> ese cliente hace login -> crea prestamo.</li>
 * </ul>
 */
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class AuthFlowIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    private ObjectMapper jsonMapper() {
        ObjectMapper m = objectMapper.copy();
        m.registerModule(new JavaTimeModule());
        return m;
    }

    private String loginAndGetToken(String email, String password) throws Exception {
        LoginRequest req = new LoginRequest(email, password);
        String body = mockMvc.perform(post("/api/auth/login")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isOk())
                .andReturn().getResponse().getContentAsString();
        return objectMapper.readValue(body, LoginResponse.class).getToken();
    }

    @Test
    @DisplayName("Admin precargado puede hacer login con password 'password'")
    void login_AdminPrecargado() throws Exception {
        LoginRequest req = new LoginRequest("admin@sgplab.edu.co", "password");
        mockMvc.perform(post("/api/auth/login")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.token").isNotEmpty())
                .andExpect(jsonPath("$.rol").value("ADMINISTRADOR"))
                .andExpect(jsonPath("$.email").value("admin@sgplab.edu.co"));
    }

    @Test
    @DisplayName("Cliente precargado puede hacer login con password 'cliente123'")
    void login_ClientePrecargado() throws Exception {
        LoginRequest req = new LoginRequest("cliente@sgplab.edu.co", "cliente123");
        mockMvc.perform(post("/api/auth/login")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.rol").value("CLIENTE"));
    }

    @Test
    @DisplayName("Login con password incorrecta -> 401")
    void login_PasswordIncorrecta() throws Exception {
        LoginRequest req = new LoginRequest("admin@sgplab.edu.co", "wrong");
        mockMvc.perform(post("/api/auth/login")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isUnauthorized());
    }

    @Test
    @DisplayName("Login con email inexistente -> 401")
    void login_EmailInexistente() throws Exception {
        LoginRequest req = new LoginRequest("nadie@nadie.com", "x");
        mockMvc.perform(post("/api/auth/login")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isUnauthorized());
    }

    @Test
    @DisplayName("Sin token, /api/usuarios -> 401")
    void sinToken_Unauthorized() throws Exception {
        mockMvc.perform(get("/api/usuarios"))
                .andExpect(status().isUnauthorized());
    }

    @Test
    @DisplayName("Con token de CLIENTE, /api/usuarios (admin) -> 403")
    void clienteNoAccedeAdmin() throws Exception {
        String token = loginAndGetToken("cliente@sgplab.edu.co", "cliente123");
        mockMvc.perform(get("/api/usuarios").header("Authorization", "Bearer " + token))
                .andExpect(status().isForbidden());
    }

    @Test
    @DisplayName("Con token de ADMIN, /api/usuarios -> 200 con lista")
    void adminAccedeUsuarios() throws Exception {
        String token = loginAndGetToken("admin@sgplab.edu.co", "password");
        mockMvc.perform(get("/api/usuarios").header("Authorization", "Bearer " + token))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray());
    }

    @Test
    @DisplayName("Token alterado -> 401")
    void tokenAlterado() throws Exception {
        String token = loginAndGetToken("admin@sgplab.edu.co", "password");
        String bad = token.substring(0, token.length() - 4) + "XXXX";
        mockMvc.perform(get("/api/usuarios").header("Authorization", "Bearer " + bad))
                .andExpect(status().isUnauthorized());
    }

    @Test
    @DisplayName("La respuesta de usuarios NUNCA expone passwordHash")
    void responsesNoExponenPassword() throws Exception {
        String token = loginAndGetToken("admin@sgplab.edu.co", "password");
        mockMvc.perform(get("/api/usuarios").header("Authorization", "Bearer " + token))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].passwordHash").doesNotExist())
                .andExpect(jsonPath("$[0].password").doesNotExist());
    }

    @Test
    @DisplayName("Flujo end-to-end: admin crea usuario, ese usuario hace login y crea un prestamo")
    void flujoCompletoLoginYPrestamo() throws Exception {
        // 1. Admin se loguea
        String adminToken = loginAndGetToken("admin@sgplab.edu.co", "password");

        // 2. Admin crea un nuevo cliente
        UsuarioRequest nuevo = new UsuarioRequest();
        nuevo.setNombre("Maria");
        nuevo.setEmail("maria@x.com");
        nuevo.setPassword("maria1234");
        nuevo.setRol(Rol.CLIENTE);

        String createdBody = mockMvc.perform(post("/api/usuarios")
                        .header("Authorization", "Bearer " + adminToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(nuevo)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.email").value("maria@x.com"))
                .andReturn().getResponse().getContentAsString();
        Long mariaId = objectMapper.readTree(createdBody).get("id").asLong();

        // 3. Admin crea un equipo nuevo
        EquipoRequest eq = new EquipoRequest();
        eq.setCodigoInventario("ITG-IT");
        eq.setNombre("Equipo integracion");
        eq.setCantidad(2);
        String eqBody = mockMvc.perform(post("/api/equipos")
                        .header("Authorization", "Bearer " + adminToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(eq)))
                .andExpect(status().isCreated())
                .andReturn().getResponse().getContentAsString();
        Long equipoId = objectMapper.readTree(eqBody).get("id").asLong();

        // 4. Maria se loguea con la password que le pusieron
        String mariaToken = loginAndGetToken("maria@x.com", "maria1234");

        // 5. Maria crea un prestamo
        PrestamoRequest pr = new PrestamoRequest();
        pr.setFechaInicio(LocalDate.now());
        pr.setFechaFin(LocalDate.now().plusDays(3));
        pr.setEquipoId(equipoId);
        pr.setUsuarioId(mariaId);

        mockMvc.perform(post("/api/prestamos")
                        .header("Authorization", "Bearer " + mariaToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(jsonMapper().writeValueAsString(pr)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.estado").value("ACTIVO"));

        // 6. Limpieza: admin elimina lo creado (en orden inverso)
        mockMvc.perform(delete("/api/usuarios/" + mariaId)
                .header("Authorization", "Bearer " + adminToken));
    }
}
