package com.corhuila.gestionpruebas.service;

import com.corhuila.gestionpruebas.model.Duenio;
import com.corhuila.gestionpruebas.model.Mascota;
import com.corhuila.gestionpruebas.repository.CitaRepository;
import com.corhuila.gestionpruebas.repository.DuenioRepository;
import com.corhuila.gestionpruebas.repository.MascotaRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class DuenioServiceTest {

    @Mock
    private DuenioRepository duenioRepository;

    // ✅ NUEVO: mocks de los repositorios agregados en el servicio
    @Mock
    private MascotaRepository mascotaRepository;

    @Mock
    private CitaRepository citaRepository;

    @InjectMocks
    private DuenioService duenioService;

    private Duenio duenio;

    @BeforeEach
    void setUp() {
        duenio = new Duenio();
        duenio.setId(1L);
        duenio.setNombre("Carlos López");
        duenio.setCorreo("carlos@email.com");
        duenio.setTelefono("3001234567");
        duenio.setDireccion("Calle 5 #10-20, Neiva");
    }

    @Test
    void guardar_conNombreValido_debeRetornarDuenio() {
        when(duenioRepository.save(duenio)).thenReturn(duenio);
        Duenio resultado = duenioService.guardar(duenio);
        assertNotNull(resultado);
        assertEquals("Carlos López", resultado.getNombre());
        verify(duenioRepository, times(1)).save(duenio);
    }

    @Test
    void guardar_conNombreNulo_debeLanzarExcepcion() {
        duenio.setNombre(null);
        assertThrows(IllegalArgumentException.class, () -> duenioService.guardar(duenio));
        verify(duenioRepository, never()).save(any());
    }

    @Test
    void guardar_conNombreVacio_debeLanzarExcepcion() {
        duenio.setNombre("");
        assertThrows(IllegalArgumentException.class, () -> duenioService.guardar(duenio));
        verify(duenioRepository, never()).save(any());
    }

    @Test
    void obtenerTodos_debeRetornarListaDuenios() {
        when(duenioRepository.findAll()).thenReturn(Arrays.asList(duenio));
        List<Duenio> resultado = duenioService.obtenerTodos();
        assertEquals(1, resultado.size());
        verify(duenioRepository, times(1)).findAll();
    }

    @Test
    void buscarPorId_conIdExistente_debeRetornarDuenio() {
        when(duenioRepository.findById(1L)).thenReturn(Optional.of(duenio));
        Duenio resultado = duenioService.buscarPorId(1L);
        assertNotNull(resultado);
        assertEquals(1L, resultado.getId());
    }

    @Test
    void buscarPorId_conIdInexistente_debeRetornarNull() {
        when(duenioRepository.findById(99L)).thenReturn(Optional.empty());
        Duenio resultado = duenioService.buscarPorId(99L);
        assertNull(resultado);
    }

    @Test
    void eliminar_conIdValido_debeEjecutarDelete() {
        // ✅ CORRECCIÓN: mockear la cascada de eliminación
        when(mascotaRepository.findByDuenioId(1L)).thenReturn(Collections.emptyList());
        doNothing().when(duenioRepository).deleteById(1L);

        duenioService.eliminar(1L);

        verify(mascotaRepository, times(1)).findByDuenioId(1L);
        verify(duenioRepository, times(1)).deleteById(1L);
    }
}