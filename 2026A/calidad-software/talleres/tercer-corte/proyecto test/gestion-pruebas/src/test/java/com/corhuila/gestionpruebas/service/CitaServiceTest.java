package com.corhuila.gestionpruebas.service;

import com.corhuila.gestionpruebas.model.Cita;
import com.corhuila.gestionpruebas.model.Mascota;
import com.corhuila.gestionpruebas.repository.CitaRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class CitaServiceTest {

    @Mock
    private CitaRepository citaRepository;

    @InjectMocks
    private CitaService citaService;

    private Cita cita;

    @BeforeEach
    void setUp() {
        Mascota mascota = new Mascota();
        mascota.setId(1L);
        mascota.setNombre("Max");

        cita = new Cita();
        cita.setId(1L);
        cita.setMotivo("Vacunación anual");
        cita.setFecha(LocalDateTime.now().plusDays(1));
        cita.setEstado("PENDIENTE");
        cita.setMascota(mascota);
    }

    @Test
    void guardar_conDatosValidos_debeRetornarCita() {
        when(citaRepository.save(cita)).thenReturn(cita);
        Cita resultado = citaService.guardar(cita);
        assertNotNull(resultado);
        assertEquals("Vacunación anual", resultado.getMotivo());
        verify(citaRepository, times(1)).save(cita);
    }

    @Test
    void guardar_conMotivoNulo_debeLanzarExcepcion() {
        cita.setMotivo(null);
        assertThrows(IllegalArgumentException.class, () -> citaService.guardar(cita));
        verify(citaRepository, never()).save(any());
    }

    @Test
    void guardar_conMotivoVacio_debeLanzarExcepcion() {
        cita.setMotivo("");
        assertThrows(IllegalArgumentException.class, () -> citaService.guardar(cita));
    }

    @Test
    void guardar_conFechaNula_debeLanzarExcepcion() {
        cita.setFecha(null);
        assertThrows(IllegalArgumentException.class, () -> citaService.guardar(cita));
        verify(citaRepository, never()).save(any());
    }

    @Test
    void guardar_sinEstado_debeAsignarPendiente() {
        cita.setEstado(null);
        when(citaRepository.save(cita)).thenReturn(cita);
        citaService.guardar(cita);
        assertEquals("PENDIENTE", cita.getEstado());
    }

    @Test
    void obtenerTodas_debeRetornarLista() {
        when(citaRepository.findAll()).thenReturn(Arrays.asList(cita));
        List<Cita> resultado = citaService.obtenerTodas();
        assertEquals(1, resultado.size());
    }

    @Test
    void buscarPorId_conIdExistente_debeRetornarCita() {
        when(citaRepository.findById(1L)).thenReturn(Optional.of(cita));
        Cita resultado = citaService.buscarPorId(1L);
        assertNotNull(resultado);
        assertEquals("PENDIENTE", resultado.getEstado());
    }

    @Test
    void buscarPorId_conIdInexistente_debeRetornarNull() {
        when(citaRepository.findById(99L)).thenReturn(Optional.empty());
        assertNull(citaService.buscarPorId(99L));
    }

    @Test
    void cambiarEstado_conIdValido_debeActualizarEstado() {
        when(citaRepository.findById(1L)).thenReturn(Optional.of(cita));
        when(citaRepository.save(cita)).thenReturn(cita);
        Cita resultado = citaService.cambiarEstado(1L, "FINALIZADA");
        assertEquals("FINALIZADA", resultado.getEstado());
    }

    @Test
    void cambiarEstado_conIdInexistente_debeLanzarExcepcion() {
        when(citaRepository.findById(99L)).thenReturn(Optional.empty());
        assertThrows(IllegalArgumentException.class,
                () -> citaService.cambiarEstado(99L, "FINALIZADA"));
    }

    @Test
    void eliminar_debeEjecutarDelete() {
        doNothing().when(citaRepository).deleteById(1L);
        citaService.eliminar(1L);
        verify(citaRepository, times(1)).deleteById(1L);
    }
}