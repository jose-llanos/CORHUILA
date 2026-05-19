package com.autospark.migueljuliana.unitarias.controllers;

import com.autospark.migueljuliana.controllers.UserController;
import com.autospark.migueljuliana.models.Role;
import com.autospark.migueljuliana.models.User;
import com.autospark.migueljuliana.services.IUserService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.security.oauth2.resource.servlet.OAuth2ResourceServerAutoConfiguration;
import org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration;
import org.springframework.boot.autoconfigure.security.servlet.SecurityFilterAutoConfiguration;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.data.jpa.mapping.JpaMetamodelMappingContext;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import java.util.List;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(
        controllers = UserController.class,
        excludeAutoConfiguration = {
                SecurityAutoConfiguration.class,
                SecurityFilterAutoConfiguration.class,
                OAuth2ResourceServerAutoConfiguration.class
        }
)
@AutoConfigureMockMvc(addFilters = false)
class UserControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private IUserService userService;

    @MockitoBean
    private JpaMetamodelMappingContext jpaMetamodelMappingContext;

    private User crearUsuario() {
        User user = new User();
        user.setId(1L);
        user.setFullName("Juan Perez");
        user.setIdentityCard("12345678");
        user.setEmail("juan@test.com");
        user.setLicensePlate("ABC123");
        user.setPhone("3001234567");
        user.setRole(Role.CUSTOMER);
        user.setPassword("hash123");
        return user;
    }

    @Test
    void getAllUsersDebeRetornarOk() throws Exception {
        when(userService.findAll())
                .thenReturn(List.of(crearUsuario()));

        mockMvc.perform(get("/autospark/users"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].email").value("juan@test.com"))
                .andExpect(jsonPath("$[0].fullName").value("Juan Perez"));

        verify(userService).findAll();
    }

    @Test
    void getUserByIdDebeRetornarOk() throws Exception {
        when(userService.findById(1L))
                .thenReturn(Optional.of(crearUsuario()));

        mockMvc.perform(get("/autospark/users/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.email").value("juan@test.com"));

        verify(userService).findById(1L);
    }

    @Test
    void registerUserDebeRetornarCreated() throws Exception {
        User user = crearUsuario();

        when(userService.existsByEmail("juan@test.com"))
                .thenReturn(false);

        when(userService.save(any(User.class)))
                .thenReturn(user);

        doNothing().when(userService)
                .sendEmail(anyString(), anyString(), anyString());

        mockMvc.perform(post("/autospark/users")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "fullName": "Juan Perez",
                                  "identityCard": "12345678",
                                  "email": "juan@test.com",
                                  "licensePlate": "ABC123",
                                  "phone": "3001234567",
                                  "role": "CUSTOMER",
                                  "password": "Password123"
                                }
                                """))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.email").value("juan@test.com"));

        verify(userService).existsByEmail("juan@test.com");
        verify(userService).save(any(User.class));
    }

    @Test
    void updateUserDebeRetornarOk() throws Exception {
        User existing = crearUsuario();

        User updated = crearUsuario();
        updated.setFullName("Juan Actualizado");

        when(userService.findById(1L))
                .thenReturn(Optional.of(existing));

        when(userService.hashPassword("NuevaPassword123"))
                .thenReturn("nuevoHash");

        when(userService.save(any(User.class)))
                .thenReturn(updated);

        mockMvc.perform(put("/autospark/users/1")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "fullName": "Juan Actualizado",
                                  "identityCard": "12345678",
                                  "email": "juan@test.com",
                                  "licensePlate": "ABC123",
                                  "phone": "3001234567",
                                  "role": "CUSTOMER",
                                  "password": "NuevaPassword123"
                                }
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.fullName").value("Juan Actualizado"));

        verify(userService).findById(1L);
        verify(userService).hashPassword("NuevaPassword123");
        verify(userService).save(any(User.class));
    }

    @Test
    void deleteUserDebeRetornarNoContent() throws Exception {
        when(userService.findById(1L))
                .thenReturn(Optional.of(crearUsuario()));

        doNothing().when(userService).delete(1L);

        mockMvc.perform(delete("/autospark/users/1"))
                .andExpect(status().isNoContent());

        verify(userService).delete(1L);
    }

    @Test
    void loginDebeRetornarOk() throws Exception {
        User user = crearUsuario();


        when(userService.login("juan@test.com", "Password123"))
                .thenReturn(Optional.of(user));

        mockMvc.perform(post("/autospark/login")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                    {
                      "email": "juan@test.com",
                      "password": "Password123"
                    }
                    """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.email").value("juan@test.com"));


        verify(userService).login("juan@test.com", "Password123");
    }

    @Test
    void recoverPasswordDebeRetornarOk() throws Exception {
        User user = crearUsuario();

        when(userService.findByEmail("juan@test.com"))
                .thenReturn(user);

        when(userService.generateRandomPassword())
                .thenReturn("Nueva123");

        when(userService.save(any(User.class)))
                .thenReturn(user);

        doNothing().when(userService)
                .sendEmail(anyString(), anyString(), anyString());

        mockMvc.perform(post("/autospark/recover-password")
                        .param("email", "juan@test.com"))
                .andExpect(status().isOk())
                .andExpect(content().string("A new password has been sent to your email"));

        verify(userService).findByEmail("juan@test.com");
        verify(userService).generateRandomPassword();
        verify(userService).save(any(User.class));
    }

    @Test
    void registerUserConEmailExistenteDebeLanzarExcepcion() throws Exception {
        when(userService.existsByEmail("juan@test.com"))
                .thenReturn(true);

        // No es necesario mockear save porque el controlador lanza la excepción antes
        mockMvc.perform(post("/autospark/users")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "fullName": "Juan Perez",
                                  "identityCard": "12345678",
                                  "email": "juan@test.com",
                                  "licensePlate": "ABC123",
                                  "phone": "3001234567",
                                  "role": "CUSTOMER",
                                  "password": "Password123"
                                }
                                """))
                .andExpect(status().isConflict());

        verify(userService).existsByEmail("juan@test.com");
        verify(userService, never()).save(any(User.class));
    }

    @Test
    void changeUserRoleDebeRetornarOk() throws Exception {
        doNothing().when(userService).changeRole(1L, Role.ADMIN);

        mockMvc.perform(put("/autospark/users/1/role")
                        .param("role", "ADMIN"))
                .andExpect(status().isOk())
                .andExpect(content().string("Rol actualizado correctamente"));

        verify(userService).changeRole(1L, Role.ADMIN);
    }

    @Test
    void loginConCredencialesInvalidasDebeRetornarUnauthorized() throws Exception {
        when(userService.login("juan@test.com", "mala"))
                .thenReturn(Optional.empty());

        mockMvc.perform(post("/autospark/login")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                    {
                      "email": "juan@test.com",
                      "password": "mala"
                    }
                    """))
                .andExpect(status().isUnauthorized())
                .andExpect(content().string("Invalid credentials"));

        verify(userService).login("juan@test.com", "mala");
    }

    @Test
    void recoverPasswordConEmailNoEncontradoDebeRetornarNotFound() throws Exception {
        when(userService.findByEmail("noexiste@test.com"))
                .thenReturn(null);

        mockMvc.perform(post("/autospark/recover-password")
                        .param("email", "noexiste@test.com"))
                .andExpect(status().isNotFound())
                .andExpect(content().string("Email not found in our system"));

        verify(userService).findByEmail("noexiste@test.com");
        verify(userService, never()).generateRandomPassword();
        verify(userService, never()).save(any(User.class));
    }

    @ParameterizedTest
    @CsvSource({
            "'', 'juan@test.com', '3001234567'",
            "'Juan Perez', 'correo-invalido', '3001234567'",
            "'Juan Perez', 'juan@test.com', '123'"
    })
    void updateUserConDatosInvalidosDebeRetornarBadRequest(
            String fullName,
            String email,
            String phone
    ) throws Exception {

        String requestBody = """
            {
              "fullName": "%s",
              "identityCard": "12345678",
              "email": "%s",
              "licensePlate": "ABC123",
              "phone": "%s",
              "role": "CUSTOMER",
              "password": "Password123"
            }
            """.formatted(fullName, email, phone);

        mockMvc.perform(put("/autospark/users/1")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestBody))
                .andExpect(status().isBadRequest());

        verify(userService, never()).findById(anyLong());
        verify(userService, never()).save(any(User.class));
    }
}