package com.map.parking_project.services;

import com.map.parking_project.models.Reservas;
import com.map.parking_project.repositories.IReservasRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ReservaServiceImplTest {

    @Mock
    private IReservasRepository reservaRepository;

    @InjectMocks
    private ReservaServiceImpl reservaService;

    private Reservas reserva;

    @BeforeEach
    void setUp() {
        reserva = new Reservas();
        reserva.setId(1L);
        // Agrega campos como reserva.setFecha() si existen en tu modelo
    }

    @Test
    void testFindAll() {
        when(reservaRepository.findAll()).thenReturn(Arrays.asList(reserva));
        assertNotNull(reservaService.findAll());
        verify(reservaRepository).findAll();
    }

    @Test
    void testSave() {
        when(reservaRepository.save(any(Reservas.class))).thenReturn(reserva);
        Reservas saved = reservaService.save(new Reservas());
        assertNotNull(saved);
        assertEquals(1L, saved.getId());
    }

    @Test
    void testDelete() {
        doNothing().when(reservaRepository).deleteById(1L);
        reservaService.delete(1L);
        verify(reservaRepository, times(1)).deleteById(1L);
    }
    @Test
    void testUpdate_Success() {
        Reservas existente = new Reservas();
        existente.setId(1L);
        when(reservaRepository.findById(1L)).thenReturn(Optional.of(existente));
        when(reservaRepository.save(any(Reservas.class))).thenReturn(existente);

        Reservas nuevosDatos = new Reservas();
        nuevosDatos.setTipo_vehiculo("Camioneta");

        reservaService.update(nuevosDatos, 1L);

        verify(reservaRepository).save(argThat(r -> r.getTipo_vehiculo().equals("Camioneta")));
    }

    @Test
    void testUpdate_NotFound() {
        when(reservaRepository.findById(99L)).thenReturn(Optional.empty());
        // Esto cubrirá la línea del System.out.println y el else
        reservaService.update(new Reservas(), 99L);
        verify(reservaRepository, never()).save(any());
    }
}