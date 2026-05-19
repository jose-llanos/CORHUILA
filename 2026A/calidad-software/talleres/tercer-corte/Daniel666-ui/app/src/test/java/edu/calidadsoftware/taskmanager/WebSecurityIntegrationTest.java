package edu.calidadsoftware.taskmanager;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestBuilders.formLogin;
import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.csrf;
import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.httpBasic;
import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.user;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.redirectedUrl;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.redirectedUrlPattern;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Pruebas de integración con MockMvc.
 *
 * Objetivo: incrementar cobertura real ejecutando el stack web (Security + MVC + API)
 * sin depender de un navegador.
 */
@SpringBootTest
@AutoConfigureMockMvc
@DisplayName("Integración Web/Security")
class WebSecurityIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    @DisplayName("El contexto de Spring Boot carga correctamente")
    void contextLoads() {
        // Si el contexto no carga, Spring lanzará excepción y la prueba fallará automáticamente.
    }

    @Nested
    @DisplayName("UI")
    class UiTests {

        @Test
        @DisplayName("GET /login responde 200")
        void loginPage_ok() throws Exception {
            mockMvc.perform(get("/login"))
                    .andExpect(status().isOk())
                    .andExpect(content().contentTypeCompatibleWith(MediaType.TEXT_HTML));
        }

        @Test
        @DisplayName("GET /dashboard sin autenticación redirige a /login")
        void dashboard_unauth_redirects() throws Exception {
            mockMvc.perform(get("/dashboard").accept(MediaType.TEXT_HTML))
                    .andExpect(status().is3xxRedirection())
                    .andExpect(redirectedUrlPattern("**/login"));
        }

        @Test
        @DisplayName("Login con credenciales válidas redirige al dashboard")
        void formLogin_success() throws Exception {
            mockMvc.perform(formLogin().user("admin").password("admin"))
                    .andExpect(status().is3xxRedirection())
                    .andExpect(redirectedUrl("/dashboard"));
        }

        @Test
        @DisplayName("POST /tasks con título inválido retorna el formulario con 200 (errores de validación)")
        void createTask_invalidTitle_returnsForm() throws Exception {
            mockMvc.perform(post("/tasks")
                            .with(user("admin").roles("ADMIN"))
                            .with(csrf())
                            .param("title", " ")
                            .param("description", "desc")
                            .param("status", "PENDING")
                            .param("priority", "MEDIUM"))
                    .andExpect(status().isOk())
                    .andExpect(content().contentTypeCompatibleWith(MediaType.TEXT_HTML));
        }
    }

    @Nested
    @DisplayName("API")
    class ApiTests {

        @Test
        @DisplayName("GET /api/tasks sin auth responde 401")
        void apiTasks_unauth_401() throws Exception {
            mockMvc.perform(get("/api/tasks"))
                    .andExpect(status().isUnauthorized());
        }

        @Test
        @DisplayName("GET /api/tasks con HTTP Basic válido responde 200 y retorna JSON")
        void apiTasks_auth_200() throws Exception {
            mockMvc.perform(get("/api/tasks").with(httpBasic("admin", "admin")))
                    .andExpect(status().isOk())
                    .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
                    .andExpect(jsonPath("$").isArray());
        }

        @Test
        @DisplayName("POST /api/users/register con email inválido responde 400 con detalle de validación")
        void register_invalidEmail_400() throws Exception {
            String body = "{\"username\":\"u1\",\"email\":\"bad\",\"password\":\"pass1234\"}";

            mockMvc.perform(post("/api/users/register")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(body))
                    .andExpect(status().isBadRequest())
                    .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
                    .andExpect(jsonPath("$.message").value("Validation failed"))
                    .andExpect(jsonPath("$.details.email").exists());
        }

        @Test
        @DisplayName("GET /api/users con USER responde 403 (endpoint ADMIN)")
        void listUsers_user_forbidden() throws Exception {
            mockMvc.perform(get("/api/users").with(httpBasic("user", "user")))
                    .andExpect(status().isForbidden());
        }
    }
}
