package com.map.parking_project.Controller;

import com.map.parking_project.controllers.TarifaRestController;
import com.map.parking_project.models.Tarifa;
import com.map.parking_project.services.ITarifaServices;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import java.util.Arrays;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(TarifaRestController.class)
class TarifaRestControllerTest {

    @Autowired
    private MockMvc mockMvc;


    @MockitoBean
    private ITarifaServices tarifaService;

    private Tarifa tarifa;

    @BeforeEach
    void setUp() {
        tarifa = new Tarifa();
        tarifa.setId(1L);
        tarifa.setTipoVehiculo("Automóvil");
        tarifa.setTarifaDiurna(3.5);
        tarifa.setTarifaNocturna(5.0);
    }

    @Test
    void testGetAllTarifas() throws Exception {
        when(tarifaService.findAll()).thenReturn(Arrays.asList(tarifa));
        mockMvc.perform(get("/api/tarifas"))
                .andExpect(status().isOk());
    }

    @Test
    void testGetTarifaById_Found() throws Exception {
        when(tarifaService.findById(1L)).thenReturn(tarifa);
        mockMvc.perform(get("/api/tarifas/1"))
                .andExpect(status().isOk());
    }

    @Test
    void testGetTarifaById_NotFound() throws Exception {
        when(tarifaService.findById(99L)).thenReturn(null);
        mockMvc.perform(get("/api/tarifas/99"))
                .andExpect(status().isNotFound());
    }

    @Test
    void testCrearTarifa_Exito() throws Exception {
        when(tarifaService.save(any(Tarifa.class))).thenReturn(tarifa);
        String json = "{\"tipoVehiculo\":\"Automóvil\", \"tarifaDiurna\":3.5, \"tarifaNocturna\":5.0}";

        mockMvc.perform(post("/api/tarifas")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(json))
                .andExpect(status().isCreated());
    }

    @Test
    void testCrearTarifa_ErrorValidacion() throws Exception {
        // Falta tarifaDiurna para forzar la excepción
        String json = "{\"tipoVehiculo\":\"Automóvil\", \"tarifaNocturna\":5.0}";

        mockMvc.perform(post("/api/tarifas")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(json))
                .andExpect(status().isBadRequest());
    }

    @Test
    void testUpdateTarifa_NotFound() throws Exception {
        when(tarifaService.findById(99L)).thenReturn(null);
        mockMvc.perform(put("/api/tarifas/99")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"tipoVehiculo\":\"Moto\"}"))
                .andExpect(status().isNotFound());
    }

    @Test
    void testDeleteTarifa() throws Exception {
        mockMvc.perform(delete("/api/tarifas/1"))
                .andExpect(status().isNoContent());
    }






}