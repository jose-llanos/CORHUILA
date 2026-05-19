package com.map.parking_project.services;

import com.map.parking_project.models.Tarifa;
import com.map.parking_project.repositories.ITarifaRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class TarifaServicesTest {

    @Mock
    private ITarifaRepository tarifaRepository;

    @InjectMocks
    private TarifaServices tarifaService;

    private Tarifa tarifa;

    @BeforeEach
    void setUp() {
        tarifa = new Tarifa();
        tarifa.setId(1L);
        // Ajusta estos campos según tu modelo Tarifa (ej. valor, tipoVehiculo)
    }

    @Test
    void testFindAll() {
        when(tarifaRepository.findAll()).thenReturn(Arrays.asList(tarifa));
        List<Tarifa> result = tarifaService.findAll();
        assertFalse(result.isEmpty());
        assertEquals(1, result.size());
    }

    @Test
    void testFindById_Found() {
        when(tarifaRepository.findById(1L)).thenReturn(Optional.of(tarifa));
        Tarifa result = tarifaService.findById(1L);
        assertNotNull(result);
        assertEquals(1L, result.getId());
    }

    @Test
    void testFindById_NotFound() {
        when(tarifaRepository.findById(2L)).thenReturn(Optional.empty());
        Tarifa result = tarifaService.findById(2L);
        assertNull(result);
    }

    @Test
    void testSave() {
        when(tarifaRepository.save(any(Tarifa.class))).thenReturn(tarifa);
        Tarifa saved = tarifaService.save(new Tarifa());
        assertNotNull(saved);
        verify(tarifaRepository, times(1)).save(any(Tarifa.class));
    }

    @Test
    void testDelete() {
        doNothing().when(tarifaRepository).deleteById(1L);
        tarifaService.delete(1L);
        verify(tarifaRepository, times(1)).deleteById(1L);
    }
}