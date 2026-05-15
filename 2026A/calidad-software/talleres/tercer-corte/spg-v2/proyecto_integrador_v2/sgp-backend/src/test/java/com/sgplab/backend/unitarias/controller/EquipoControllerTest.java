package com.sgplab.backend.unitarias.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sgplab.backend.controller.EquipoController;
import com.sgplab.backend.dto.request.EquipoRequest;
import com.sgplab.backend.dto.response.EquipoResponse;
import com.sgplab.backend.exception.BusinessRuleException;
import com.sgplab.backend.exception.GlobalExceptionHandler;
import com.sgplab.backend.model.enums.EstadoEquipo;
import com.sgplab.backend.service.contract.IEquipoService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class EquipoControllerTest {

    private MockMvc mockMvc;
    private IEquipoService equipoService;
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        equipoService = Mockito.mock(IEquipoService.class);
        EquipoController controller = new EquipoController(equipoService);
        mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
        objectMapper = new ObjectMapper();
    }

    @Test
    @DisplayName("GET /api/equipos OK")
    void obtenerTodos() throws Exception {
        Mockito.when(equipoService.obtenerTodos()).thenReturn(List.of(
                new EquipoResponse(1L, "MIC-001", "Microscopio", 5, EstadoEquipo.DISPONIBLE)
        ));
        mockMvc.perform(get("/api/equipos"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].codigoInventario").value("MIC-001"));
    }

    @Test
    @DisplayName("GET /api/equipos/{id} OK")
    void obtenerPorId_OK() throws Exception {
        Mockito.when(equipoService.obtenerPorId(1L))
                .thenReturn(new EquipoResponse(1L, "MIC-001", "Microscopio", 5, EstadoEquipo.DISPONIBLE));
        mockMvc.perform(get("/api/equipos/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.cantidad").value(5));
    }

    @Test
    @DisplayName("POST /api/equipos -> 201")
    void crear_OK() throws Exception {
        EquipoRequest req = new EquipoRequest();
        req.setCodigoInventario("ESP-100");
        req.setNombre("Espectro");
        req.setCantidad(1);

        Mockito.when(equipoService.crear(any()))
                .thenReturn(new EquipoResponse(9L, "ESP-100", "Espectro", 1, EstadoEquipo.DISPONIBLE));

        mockMvc.perform(post("/api/equipos")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").value(9));
    }

    @Test
    @DisplayName("POST /api/equipos -> 400 cantidad negativa")
    void crear_CantidadNegativa() throws Exception {
        EquipoRequest req = new EquipoRequest();
        req.setCodigoInventario("X");
        req.setNombre("X");
        req.setCantidad(-1);
        mockMvc.perform(post("/api/equipos")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isBadRequest());
    }

    @Test
    @DisplayName("DELETE /api/equipos/{id} OK -> 204")
    void eliminar_OK() throws Exception {
        mockMvc.perform(delete("/api/equipos/1"))
                .andExpect(status().isNoContent());
        Mockito.verify(equipoService).eliminar(1L);
    }

    @Test
    @DisplayName("DELETE /api/equipos/{id} -> 409 cuando hay prestamos activos")
    void eliminar_Conflicto() throws Exception {
        Mockito.doThrow(new BusinessRuleException("Tiene prestamos activos"))
                .when(equipoService).eliminar(1L);
        mockMvc.perform(delete("/api/equipos/1"))
                .andExpect(status().isConflict());
    }
}
