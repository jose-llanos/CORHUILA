package com.corhuila.gestionpruebas.service;

import com.corhuila.gestionpruebas.model.Cita;
import com.corhuila.gestionpruebas.model.Tratamiento;
import com.corhuila.gestionpruebas.repository.TratamientoRepository;
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
class TratamientoServiceTest {

    @Mock
    private TratamientoRepository tratamientoRepository;

    @InjectMocks
    private TratamientoService tratamientoService;

    private Tratamiento tratamiento;

    @BeforeEach
    void setUp() {
        Cita cita = new Cita();
        cita.setId(1L);
        cita.setMotivo("Vacunación");

        tratamiento = new Tratamiento();
        tratamiento.setId(1L);
        tratamiento.setDescripcion("Aplicación vacuna antirrábica");
        tratamiento.setMedicamento("Nobivac Rabies");
        tratamiento.setDosis("1ml subcutáneo");
        tratamiento.setCita(cita);
    }

    @Test
    void guardar_conDatosValidos_debeRetornarTratamiento() {
        when(tratamientoRepository.save(tratamiento)).thenReturn(tratamiento);
        Tratamiento resultado = tratamientoService.guardar(tratamiento);
        assertNotNull(resultado);
        assertEquals("Aplicación vacuna antirrábica", resultado.getDescripcion());
        verify(tratamientoRepository, times(1)).save(tratamiento);
    }

    @Test
    void guardar_conDescripcionNula_debeLanzarExcepcion() {
        tratamiento.setDescripcion(null);
        assertThrows(IllegalArgumentException.class,
                () -> tratamientoService.guardar(tratamiento));
        verify(tratamientoRepository, never()).save(any());
    }

    @Test
    void guardar_conDescripcionVacia_debeLanzarExcepcion() {
        tratamiento.setDescripcion("");
        assertThrows(IllegalArgumentException.class,
                () -> tratamientoService.guardar(tratamiento));
    }

    @Test
    void obtenerTodos_debeRetornarLista() {
        when(tratamientoRepository.findAll()).thenReturn(Arrays.asList(tratamiento));
        List<Tratamiento> resultado = tratamientoService.obtenerTodos();
        assertEquals(1, resultado.size());
        assertEquals("Nobivac Rabies", resultado.get(0).getMedicamento());
    }

    @Test
    void buscarPorId_conIdExistente_debeRetornarTratamiento() {
        when(tratamientoRepository.findById(1L)).thenReturn(Optional.of(tratamiento));
        Tratamiento resultado = tratamientoService.buscarPorId(1L);
        assertNotNull(resultado);
        assertEquals("1ml subcutáneo", resultado.getDosis());
    }

    @Test
    void buscarPorId_conIdInexistente_debeRetornarNull() {
        when(tratamientoRepository.findById(99L)).thenReturn(Optional.empty());
        assertNull(tratamientoService.buscarPorId(99L));
    }

    @Test
    void eliminar_debeEjecutarDelete() {
        doNothing().when(tratamientoRepository).deleteById(1L);
        tratamientoService.eliminar(1L);
        verify(tratamientoRepository, times(1)).deleteById(1L);
    }
}