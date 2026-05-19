package com.map.parking_project.Controller;

import com.map.parking_project.controllers.UserRestController;
import com.map.parking_project.models.User;
import com.map.parking_project.services.IUserService;
import jakarta.mail.MessagingException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import java.util.Arrays;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(UserRestController.class)
class UserRestControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private IUserService userService;

    private User user;

    @BeforeEach
    void setUp() {
        user = new User();
        user.setId(1L);
        user.setName("Juan");
        user.setEmail("juan@test.com");
        user.setPlate("ABC-123");
        user.setTypecar("Automóvil");
    }

    @Test
    void testIndex_ReturnsList() throws Exception {
        when(userService.findAll()).thenReturn(Arrays.asList(user));

        mockMvc.perform(get("/api/user"))
                .andExpect(status().isOk())
                .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$[0].name").value("Juan"));
    }

    @Test
    void testShow_UserExists() throws Exception {
        when(userService.findById(1L)).thenReturn(user);

        mockMvc.perform(get("/api/user/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.email").value("juan@test.com"));
    }

    @Test
    void testShow_UserNotFound() throws Exception {
        when(userService.findById(anyLong())).thenReturn(null);

        mockMvc.perform(get("/api/user/99"))
                .andExpect(status().isNotFound());
    }

    @Test
    void testRecuperarContrasenia_Success() throws Exception {
        when(userService.findByEmail("juan@test.com")).thenReturn(user);
        when(userService.generarContraseniaAleatoria()).thenReturn("NewPass123");

        mockMvc.perform(post("/api/recuperarcontrasenia").param("email", "juan@test.com"))
                .andExpect(status().isOk())
                .andExpect(content().json("{\"message\":\"Se ha enviado un correo con la nueva contraseña.\"}"));
    }

    @Test
    void testValidarYCalcularTarifa_Success() throws Exception {
        // Objeto de petición simulado
        String jsonRequest = "{\"plate\":\"ABC-123\", \"typecar\":\"Automóvil\", \"hours\":2}";

        // Mock de un usuario que tiene 'hours' (asumiendo que tu modelo User tiene ese campo o similar)
        user.setHours(2);
        when(userService.findByPlate("ABC-123")).thenReturn(user);

        mockMvc.perform(post("/api/validar-tarifa")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(jsonRequest))
                .andExpect(status().isOk())
                .andExpect(content().json("{\"total\":7.0,\"message\":\"Tarifa calculada con éxito\"}")); // 3.5 * 2 horas
    }

    @Test
    void testValidarYCalcularTarifa_InvalidType() throws Exception {
        String jsonRequest = "{\"plate\":\"ABC-123\", \"typecar\":\"Moto\", \"hours\":2}";
        when(userService.findByPlate("ABC-123")).thenReturn(user); // user es 'Automóvil'

        mockMvc.perform(post("/api/validar-tarifa")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(jsonRequest))
                .andExpect(status().isBadRequest());
    }
    @Test
    void testRecuperarContraseña_ErrorEnvioEmail() throws Exception {
        when(userService.findByEmail(anyString())).thenReturn(new User());
        // Forzamos el error de correo para entrar al CATCH
        doThrow(new jakarta.mail.MessagingException("Error simulado"))
                .when(userService).sendEmail(anyString(), anyString(), anyString());

        mockMvc.perform(post("/api/recuperarcontrasenia").param("email", "test@test.com"))
                .andExpect(status().isInternalServerError()); // Esto cubre el bloque 'catch'
    }

    @Test
    void testCasosError_IfElseAndCatch() throws Exception {
        // Caso: El tipo de vehículo NO coincide (Lanza RuntimeException)
        User u = new User(); u.setTypecar("Moto");
        when(userService.findByPlate(anyString())).thenReturn(u);
        mockMvc.perform(post("/api/validar-tarifa").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"plate\":\"X\", \"typecar\":\"Automóvil\"}"))
                .andExpect(status().isBadRequest());

        // Caso: Error al enviar correo (Cubre el MessagingException catch)
        when(userService.findByEmail(anyString())).thenReturn(new User());
        doThrow(new MessagingException()).when(userService).sendEmail(anyString(), anyString(), anyString());
        mockMvc.perform(get("/api/send").param("to","a").param("subject","b").param("body","c"))
                .andExpect(content().string(org.hamcrest.Matchers.containsString("Error")));
    }

    @Test
    void testUpdateUser_Exito() throws Exception {
        // 1. Preparamos los datos
        User usuarioExistente = new User();
        usuarioExistente.setId(1L);
        usuarioExistente.setName("Original");

        User nuevosDatos = new User();
        nuevosDatos.setName("Nuevo Nombre");
        nuevosDatos.setLastname("Apellido");
        nuevosDatos.setPhone("123456");
        nuevosDatos.setPlate("ABC-123");
        nuevosDatos.setTypecar("Moto");
        nuevosDatos.setEmail("nuevo@test.com");
        nuevosDatos.setPassword("pass123");
        nuevosDatos.setRol("USER");

        // 2. Configuramos los Mocks
        org.mockito.Mockito.when(userService.findById(1L)).thenReturn(usuarioExistente);
        org.mockito.Mockito.when(userService.save(org.mockito.ArgumentMatchers.any(User.class))).thenReturn(usuarioExistente);

        // 3. Ejecutamos la petición PUT
        mockMvc.perform(org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put("/api/user/1")
                        .contentType(org.springframework.http.MediaType.APPLICATION_JSON)
                        .content("{\"name\":\"Nuevo Nombre\", \"lastname\":\"Apellido\", \"phone\":\"123456\", \"plate\":\"ABC-123\", \"typecar\":\"Moto\", \"email\":\"nuevo@test.com\", \"password\":\"pass123\", \"rol\":\"USER\"}"))
                .andExpect(status().isCreated()) // Tu código tiene HttpStatus.CREATED
                .andExpect(jsonPath("$.name").value("Nuevo Nombre"));

        // 4. Verificamos que se llamó al save
        org.mockito.Mockito.verify(userService, org.mockito.Mockito.times(1)).save(org.mockito.ArgumentMatchers.any(User.class));
    }

    @Test
    void testUpdateUser_NoEncontrado() throws Exception {
        // Simulamos que el service devuelve null
        org.mockito.Mockito.when(userService.findById(99L)).thenReturn(null);

        mockMvc.perform(org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put("/api/user/99")
                        .contentType(org.springframework.http.MediaType.APPLICATION_JSON)
                        .content("{\"name\":\"Nadie\"}"))
                .andExpect(status().isNotFound());
    }
}