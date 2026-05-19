package com.map.parking_project.Controller;

import com.map.parking_project.controllers.ReservaRestController;
import com.map.parking_project.models.Reservas;
import com.map.parking_project.services.IReservaService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import com.map.parking_project.dto.ReservaDTO;
import java.time.LocalDate;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Arrays;
import java.util.Optional;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(ReservaRestController.class)
class ReservaRestControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockitoBean
    private IReservaService reservaService;

    private Reservas reserva;

    @BeforeEach
    void setUp() {
        reserva = new Reservas();
        reserva.setId(1L);
        reserva.setTipo_vehiculo("Moto");
    }

    @Test
    void listarReservas() throws Exception {
        when(reservaService.findAll()).thenReturn(Arrays.asList(reserva));
        mockMvc.perform(get("/api/reservas"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].tipo_vehiculo").value("Moto"));
    }

    @Test
    void obtenerReserva_Encontrada() throws Exception {
        when(reservaService.findById(1L)).thenReturn(Optional.of(reserva));
        mockMvc.perform(get("/api/reservas/1"))
                .andExpect(status().isOk());
    }

    @Test
    void obtenerReserva_NoEncontrada() throws Exception {
        when(reservaService.findById(1L)).thenReturn(Optional.empty());
        mockMvc.perform(get("/api/reservas/1"))
                .andExpect(status().isNotFound());
    }

    @Test
    void registrarReserva_Exito() throws Exception {
        when(reservaService.save(any(Reservas.class))).thenReturn(reserva);
        mockMvc.perform(post("/api/reservas")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"tipo_vehiculo\":\"Moto\"}"))
                .andExpect(status().isCreated());
    }

    @Test
    void registrarReserva_Error() throws Exception {
        when(reservaService.save(any(Reservas.class))).thenThrow(new RuntimeException("Error"));
        mockMvc.perform(post("/api/reservas")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"tipo_vehiculo\":\"Moto\"}"))
                .andExpect(status().isInternalServerError());
    }

    @Test
    void testCondicionesInexistentes() throws Exception {
        // Caso: Buscar reserva que NO existe
        when(reservaService.findById(99L)).thenReturn(Optional.empty());
        mockMvc.perform(get("/api/reservas/99")).andExpect(status().isNotFound());

        // Caso: Eliminar reserva que NO existe (Cubre el ELSE de eliminar)
        mockMvc.perform(delete("/api/reservas/99")).andExpect(status().isNotFound());
    }

    @Test
    void testConfirmarReserva_Exito() throws Exception {
        reserva = new Reservas();
        reserva.setId(1L);
        reserva.setConfirmada(false);

        org.mockito.Mockito.when(reservaService.findById(1L)).thenReturn(java.util.Optional.of(reserva));
        org.mockito.Mockito.when(reservaService.save(org.mockito.ArgumentMatchers.any(Reservas.class))).thenReturn(reserva);

        // CAMBIO: Se usa .put() en lugar de .post()
        mockMvc.perform(org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put("/api/reservas/1/confirmar")
                        .contentType(org.springframework.http.MediaType.APPLICATION_JSON))
                .andExpect(status().isOk())
                .andExpect(content().string(org.hamcrest.Matchers.containsString("Reserva confirmada")));

        org.mockito.Mockito.verify(reservaService, org.mockito.Mockito.times(1)).save(org.mockito.ArgumentMatchers.any(Reservas.class));
    }
    @Test
    void testConfirmarReserva_NoEncontrada() throws Exception {
        org.mockito.Mockito.when(reservaService.findById(org.mockito.ArgumentMatchers.anyLong())).thenReturn(java.util.Optional.empty());

        // CAMBIO: Se usa .put() en lugar de .post()
        mockMvc.perform(org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put("/api/reservas/999/confirmar")
                        .contentType(org.springframework.http.MediaType.APPLICATION_JSON))
                .andExpect(status().isNotFound());

        org.mockito.Mockito.verify(reservaService, org.mockito.Mockito.never()).save(org.mockito.ArgumentMatchers.any(Reservas.class));
    }

    @Test
    void testActualizarReserva_Exito() throws Exception {
        // 1. Preparar los datos de prueba
        Long reservaId = 1L;
        
        // El DTO que simula lo que envía Angular
        ReservaDTO reservaDTO = new ReservaDTO();
        reservaDTO.setTipo_vehiculo("Camioneta");
        reservaDTO.setTipo_servicio("Lavado Completo");
        reservaDTO.setHoras(3);
        reservaDTO.setFecha(LocalDate.now());
        reservaDTO.setPrecio(25.0);

        // La Entidad vieja que simula lo que está en la base de datos
        Reservas reservaExistente = new Reservas();
        reservaExistente.setId(reservaId);
        reservaExistente.setTipo_vehiculo("Moto"); // Este dato debería cambiar

        // 2. Configurar el comportamiento del Mock
        when(reservaService.findById(reservaId)).thenReturn(Optional.of(reservaExistente));
        
        // 3. Ejecutar la petición simulada y verificar resultados
        mockMvc.perform(put("/api/reservas/{id}", reservaId)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(reservaDTO)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.message").value("Reserva actualizada correctamente"))
                .andExpect(jsonPath("$.reserva.tipo_vehiculo").value("Camioneta")); // Verifica que mapeó bien
    }

    @Test
    void testActualizarReserva_NoEncontrada() throws Exception {
        // 1. Preparar los datos
        Long reservaId = 99L; // Un ID que no existe
        ReservaDTO reservaDTO = new ReservaDTO();
        reservaDTO.setTipo_vehiculo("Automóvil");

        // 2. Configurar el Mock para que devuelva vacío
        when(reservaService.findById(reservaId)).thenReturn(Optional.empty());

        // 3. Ejecutar la petición simulada y verificar el error 404
        mockMvc.perform(put("/api/reservas/{id}", reservaId)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(reservaDTO)))
                .andExpect(status().isNotFound())
                // Si tienes acceso a tu constante MESSAGE_KEY, puedes descomentar la siguiente línea:
                // .andExpect(jsonPath("$.message").exists()) 
                ;
    }
}