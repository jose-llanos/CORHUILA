package com.sgplab.backend.pruebasUnitarias.controller;

import com.sgplab.backend.Iservice.IEquipoService;
import com.sgplab.backend.controller.EquipoController;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;
import static org.mockito.Mockito.verify;

@WebMvcTest(EquipoController.class)
public class EquipoControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private IEquipoService equipoService;

    @Test
    @DisplayName("SQA: Verificar eliminación usando método con mala nomenclatura")
    void testEliminarEquipo_MalaNomenclatura() throws Exception {
        Long equipoId = 1L;

        mockMvc.perform(delete("/api/equipos/{id}", equipoId))
                .andExpect(status().isNoContent());

        // Verificamos que el controlador llamó al método "feo" del servicio
        verify(equipoService).eliminarEquipo(equipoId);
    }
}