package com.sgplab.backend.pruebasUnitarias.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sgplab.backend.controller.UsuarioController;
import com.sgplab.backend.model.entity.Usuario;
import com.sgplab.backend.model.enums.Rol;
import com.sgplab.backend.Iservice.IUsuarioService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(UsuarioController.class)
public class UsuarioControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private IUsuarioService usuarioService;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    @DisplayName("SQA: Confirmar vulnerabilidad de Elevación de Privilegios vía API")
    void testCrearUsuario_AceptaRolAdminSinFiltro() throws Exception {
        Usuario hacker = new Usuario();
        hacker.setNombre("Hacker");
        hacker.setRol(Rol.ADMINISTRADOR);

        Mockito.when(usuarioService.crearUsuario(Mockito.any(Usuario.class)))
                .thenReturn(hacker);

        mockMvc.perform(post("/api/usuarios")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(hacker)))
                .andExpect(status().isCreated())
                // La prueba confirma que el rol devuelto es ADMIN, evidenciando la falta de seguridad
                .andExpect(jsonPath("$.rol").value("ADMINISTRADOR"));
    }
}
