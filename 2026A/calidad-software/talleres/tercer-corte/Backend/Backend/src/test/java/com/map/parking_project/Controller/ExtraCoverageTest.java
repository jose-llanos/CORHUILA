package com.map.parking_project.Controller;

import com.map.parking_project.controllers.VehicleEntryRestController;
import com.map.parking_project.models.VehicleEntry;
import com.map.parking_project.services.VehicleEntryService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;

import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import java.util.Arrays;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(VehicleEntryRestController.class)
class VehicleEntryRestControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private VehicleEntryService vehicleService;

    private VehicleEntry vehicleEntry;

    @BeforeEach
    void setUp() {
        vehicleEntry = new VehicleEntry();
        vehicleEntry.setId(1L);
        vehicleEntry.setPlaca("ABC-123");
    }

    @Test
    void testListarIngresos() throws Exception {
        when(vehicleService.findAll()).thenReturn(Arrays.asList(vehicleEntry));

        mockMvc.perform(get("/api/ingresos"))
                .andExpect(status().isOk())
                .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON));
    }

    @Test
    void testRegistrarIngreso() throws Exception {
        when(vehicleService.save(any(VehicleEntry.class))).thenReturn(vehicleEntry);

        String json = "{\"placa\":\"ABC-123\"}";

        mockMvc.perform(post("/api/ingresos")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(json))
                .andExpect(status().isCreated()); // ✅ ¡Cambiado a isCreated() para que espere el 201!
    }

    @Test
    void testDeleteIngreso() throws Exception {
        doNothing().when(vehicleService).delete(1L);

        mockMvc.perform(delete("/api/ingresos/1"))
                .andExpect(status().isNoContent());

        verify(vehicleService).delete(1L);
    }
}