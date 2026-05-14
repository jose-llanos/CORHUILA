package com.sgplab.backend.pruebasUnitarias.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sgplab.backend.controller.PrestamoController;
import com.sgplab.backend.model.entity.Equipo;
import com.sgplab.backend.model.entity.Prestamo;
import com.sgplab.backend.Iservice.IPrestamoService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(PrestamoController.class)
public class PrestamoControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private IPrestamoService prestamoService;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    @DisplayName("Evidencia SQA: Validación manual espagueti devuelve 200 OK en lugar de 400 Bad Request")
    void testCrearPrestamo_SinEquipo_RetornaFalsoOk() throws Exception {
        Prestamo prestamoSinEquipo = new Prestamo();
        // No le asignamos equipo para disparar la validación manual

        mockMvc.perform(post("/api/prestamos/HACER_EL_PRESTAMO_NUEVO_SISTEMA")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(prestamoSinEquipo)))
                // Comprobamos la mala práctica: Devuelve 200 OK en lugar de error de cliente
                .andExpect(status().isOk())
                .andExpect(content().string("FALTA EL EQUIPO"));
    }

    @Test
    @DisplayName("Evidencia SQA: Error crítico interno es silenciado y devuelve 200 OK engañoso")
    void testCrearPrestamo_ErrorInterno_RetornaFalsoOk() throws Exception {
        Prestamo prestamo = new Prestamo();
        prestamo.setEquipo(new Equipo()); // Pasamos la validación manual

        // Simulamos que al intentar guardar, la base de datos colapsa y lanza una excepción
        Mockito.when(prestamoService.crearPrestamo(Mockito.any(Prestamo.class)))
                .thenThrow(new RuntimeException("Conexión perdida con la Base de Datos"));

        mockMvc.perform(post("/api/prestamos/HACER_EL_PRESTAMO_NUEVO_SISTEMA")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(prestamo)))
                // Comprobamos la falla grave: El sistema oculta el error y dice que todo salió bien (200)
                .andExpect(status().isOk())
                .andExpect(content().string("Ocurrió un error pero no sé cuál"));
    }
}