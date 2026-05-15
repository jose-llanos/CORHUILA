package com.sgplab.backend.unitarias.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.sgplab.backend.controller.PenalizacionController;
import com.sgplab.backend.dto.request.PenalizacionRequest;
import com.sgplab.backend.dto.response.PenalizacionResponse;
import com.sgplab.backend.exception.GlobalExceptionHandler;
import com.sgplab.backend.model.enums.EstadoPenalizacion;
import com.sgplab.backend.service.contract.IPenalizacionService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.time.LocalDate;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class PenalizacionControllerTest {

    private MockMvc mockMvc;
    private IPenalizacionService penalizacionService;
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        penalizacionService = Mockito.mock(IPenalizacionService.class);
        PenalizacionController controller = new PenalizacionController(penalizacionService);
        mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
        objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());
    }

    @Test
    @DisplayName("GET /api/penalizaciones OK")
    void obtenerTodas() throws Exception {
        Mockito.when(penalizacionService.obtenerTodas()).thenReturn(List.of(new PenalizacionResponse()));
        mockMvc.perform(get("/api/penalizaciones")).andExpect(status().isOk());
    }

    @Test
    @DisplayName("GET /api/penalizaciones/usuario/{id}/activa true")
    void verificarPenalizacionActiva() throws Exception {
        Mockito.when(penalizacionService.tienePenalizacionActiva(5L)).thenReturn(true);
        mockMvc.perform(get("/api/penalizaciones/usuario/5/activa"))
                .andExpect(status().isOk())
                .andExpect(content().string("true"));
    }

    @Test
    @DisplayName("POST /api/penalizaciones -> 201")
    void crear_OK() throws Exception {
        PenalizacionRequest req = new PenalizacionRequest();
        req.setMotivo("Devolucion tardia");
        req.setFechaInicio(LocalDate.now());
        req.setFechaFin(LocalDate.now().plusDays(7));
        req.setUsuarioId(1L);

        PenalizacionResponse resp = new PenalizacionResponse();
        resp.setId(11L);
        resp.setEstado(EstadoPenalizacion.ACTIVA);
        Mockito.when(penalizacionService.crear(any())).thenReturn(resp);

        mockMvc.perform(post("/api/penalizaciones")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isCreated());
    }

    @Test
    @DisplayName("POST /api/penalizaciones -> 400 sin campos")
    void crear_400() throws Exception {
        PenalizacionRequest req = new PenalizacionRequest();
        mockMvc.perform(post("/api/penalizaciones")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isBadRequest());
    }
}
