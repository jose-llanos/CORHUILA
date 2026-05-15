package com.sgplab.backend.unitarias.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.sgplab.backend.controller.PrestamoController;
import com.sgplab.backend.dto.request.PrestamoRequest;
import com.sgplab.backend.dto.response.PrestamoResponse;
import com.sgplab.backend.exception.BusinessRuleException;
import com.sgplab.backend.exception.GlobalExceptionHandler;
import com.sgplab.backend.model.enums.EstadoPrestamo;
import com.sgplab.backend.service.contract.IPrestamoService;
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
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class PrestamoControllerTest {

    private MockMvc mockMvc;
    private IPrestamoService prestamoService;
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        prestamoService = Mockito.mock(IPrestamoService.class);
        PrestamoController controller = new PrestamoController(prestamoService);
        mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
        objectMapper = new ObjectMapper();
        objectMapper.registerModule(new JavaTimeModule());
    }

    private PrestamoResponse buildResponse() {
        PrestamoResponse r = new PrestamoResponse();
        r.setId(1L);
        r.setFechaInicio(LocalDate.now());
        r.setFechaFin(LocalDate.now().plusDays(5));
        r.setEquipoId(2L);
        r.setEquipoNombre("Microscopio");
        r.setUsuarioId(3L);
        r.setUsuarioNombre("Cliente");
        r.setEstado(EstadoPrestamo.ACTIVO);
        return r;
    }

    @Test
    @DisplayName("GET /api/prestamos OK")
    void obtenerTodos() throws Exception {
        Mockito.when(prestamoService.obtenerTodos()).thenReturn(List.of(buildResponse()));
        mockMvc.perform(get("/api/prestamos"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].estado").value("ACTIVO"));
    }

    @Test
    @DisplayName("GET /api/prestamos/{id} OK")
    void obtenerPorId_OK() throws Exception {
        Mockito.when(prestamoService.obtenerPorId(1L)).thenReturn(buildResponse());
        mockMvc.perform(get("/api/prestamos/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.equipoNombre").value("Microscopio"));
    }

    @Test
    @DisplayName("POST /api/prestamos -> 201")
    void crear_OK() throws Exception {
        PrestamoRequest req = new PrestamoRequest();
        req.setFechaInicio(LocalDate.now());
        req.setFechaFin(LocalDate.now().plusDays(3));
        req.setEquipoId(2L);
        req.setUsuarioId(3L);

        Mockito.when(prestamoService.crear(any())).thenReturn(buildResponse());

        mockMvc.perform(post("/api/prestamos")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").value(1));
    }

    @Test
    @DisplayName("POST /api/prestamos -> 400 cuando faltan campos")
    void crear_400() throws Exception {
        PrestamoRequest req = new PrestamoRequest();
        mockMvc.perform(post("/api/prestamos")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isBadRequest());
    }

    @Test
    @DisplayName("POST /api/prestamos -> 409 cuando regla de negocio falla")
    void crear_409() throws Exception {
        PrestamoRequest req = new PrestamoRequest();
        req.setFechaInicio(LocalDate.now());
        req.setEquipoId(2L);
        req.setUsuarioId(3L);

        Mockito.when(prestamoService.crear(any()))
                .thenThrow(new BusinessRuleException("Sin stock"));

        mockMvc.perform(post("/api/prestamos")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(req)))
                .andExpect(status().isConflict());
    }
}
