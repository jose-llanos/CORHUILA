package com.corhuila.gestionpruebas.service;

import com.corhuila.gestionpruebas.model.Duenio;
import com.corhuila.gestionpruebas.model.Mascota;
import com.corhuila.gestionpruebas.repository.CitaRepository;
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
class MascotaServiceTest {

    @Mock
    private MascotaRepository mascotaRepository;

    // ✅ NUEVO: mock del repositorio agregado en el servicio
    @Mock
    private CitaRepository citaRepository;

    @InjectMocks
    private MascotaService mascotaService;

    private Mascota mascota;

    @BeforeEach
    void setUp() {
        Duenio duenio = new Duenio();
        duenio.setId(1L);
        duenio.setNombre("Carlos López");

        mascota = new Mascota();
        mascota.setId(1L);
        mascota.setNombre("Max");
        mascota.setEspecie("Perro");
        mascota.setRaza("Labrador");
        mascota.setEdad(3);
        mascota.setPeso(25.5);
        mascota.setDuenio(duenio);
    }

    @Test
    void guardar_conDatosValidos_debeRetornarMascota() {
        when(mascotaRepository.save(mascota)).thenReturn(mascota);
        Mascota resultado = mascotaService.guardar(mascota);
        assertNotNull(resultado);
        assertEquals("Max", resultado.getNombre());
        verify(mascotaRepository, times(1)).save(mascota);
    }

    @Test
    void guardar_conNombreNulo_debeLanzarExcepcion() {
        mascota.setNombre(null);
        assertThrows(IllegalArgumentException.class, () -> mascotaService.guardar(mascota));
        verify(mascotaRepository, never()).save(any());
    }

    @Test
    void guardar_conNombreVacio_debeLanzarExcepcion() {
        mascota.setNombre("");
        assertThrows(IllegalArgumentException.class, () -> mascotaService.guardar(mascota));
    }

    @Test
    void guardar_conEspecieNula_debeLanzarExcepcion() {
        mascota.setEspecie(null);
        assertThrows(IllegalArgumentException.class, () -> mascotaService.guardar(mascota));
        verify(mascotaRepository, never()).save(any());
    }

    @Test
    void guardar_conEspecieVacia_debeLanzarExcepcion() {
        mascota.setEspecie("");
        assertThrows(IllegalArgumentException.class, () -> mascotaService.guardar(mascota));
    }

    @Test
    void obtenerTodas_debeRetornarLista() {
        when(mascotaRepository.findAll()).thenReturn(Arrays.asList(mascota));
        List<Mascota> resultado = mascotaService.obtenerTodas();
        assertEquals(1, resultado.size());
        assertEquals("Max", resultado.get(0).getNombre());
    }

    @Test
    void buscarPorId_conIdExistente_debeRetornarMascota() {
        when(mascotaRepository.findById(1L)).thenReturn(Optional.of(mascota));
        Mascota resultado = mascotaService.buscarPorId(1L);
        assertNotNull(resultado);
        assertEquals("Perro", resultado.getEspecie());
    }

    @Test
    void buscarPorId_conIdInexistente_debeRetornarNull() {
        when(mascotaRepository.findById(99L)).thenReturn(Optional.empty());
        assertNull(mascotaService.buscarPorId(99L));
    }

    @Test
    void eliminar_debeEjecutarDelete() {
        // ✅ CORRECCIÓN: mockear la cascada de eliminación
        when(citaRepository.findByMascotaId(1L)).thenReturn(Collections.emptyList());
        doNothing().when(mascotaRepository).deleteById(1L);

        mascotaService.eliminar(1L);

        verify(citaRepository, times(1)).findByMascotaId(1L);
        verify(mascotaRepository, times(1)).deleteById(1L);
    }
}