package com.sgplab.backend.pruebasUnitarias.controller;

import com.sgplab.backend.Iservice.IPenalizacionService;
import com.sgplab.backend.controller.PenalizacionController;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(PenalizacionController.class)
public class PenalizacionControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private IPenalizacionService penalizacionService;

    @Test
    @DisplayName("SQA Baseline: Verificar consulta de penalización activa (Código Limpio)")
    void testVerificarPenalizacion_UsuarioActivo() throws Exception {
        Long usuarioId = 1L;
        Mockito.when(penalizacionService.isUsuarioPenalizado(usuarioId)).thenReturn(true);

        mockMvc.perform(get("/api/penalizaciones/usuario/{id}/activa", usuarioId))
                .andExpect(status().isOk())
                .andExpect(content().string("true"));
    }
}