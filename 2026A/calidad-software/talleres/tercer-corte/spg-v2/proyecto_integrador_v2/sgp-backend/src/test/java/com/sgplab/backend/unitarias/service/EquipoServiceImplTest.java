package com.sgplab.backend.unitarias.service;

import com.sgplab.backend.dto.request.EquipoRequest;
import com.sgplab.backend.dto.response.EquipoResponse;
import com.sgplab.backend.exception.BusinessRuleException;
import com.sgplab.backend.exception.DuplicateResourceException;
import com.sgplab.backend.exception.ResourceNotFoundException;
import com.sgplab.backend.model.entity.Equipo;
import com.sgplab.backend.model.enums.EstadoEquipo;
import com.sgplab.backend.model.enums.EstadoPrestamo;
import com.sgplab.backend.repository.IEquipoRepository;
import com.sgplab.backend.repository.IPrestamoRepository;
import com.sgplab.backend.service.impl.EquipoServiceImpl;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class EquipoServiceImplTest {

    @Mock
    private IEquipoRepository equipoRepository;
    @Mock
    private IPrestamoRepository prestamoRepository;
    @InjectMocks
    private EquipoServiceImpl equipoService;

    private EquipoRequest validRequest;
    private Equipo existente;

    @BeforeEach
    void setUp() {
        validRequest = new EquipoRequest();
        validRequest.setCodigoInventario("MIC-001");
        validRequest.setNombre("Microscopio");
        validRequest.setCantidad(3);
        validRequest.setEstado(EstadoEquipo.DISPONIBLE);

        existente = new Equipo();
        existente.setId(10L);
        existente.setCodigoInventario("ESP-002");
        existente.setNombre("Espectro");
        existente.setCantidad(2);
        existente.setEstado(EstadoEquipo.DISPONIBLE);
    }

    @Test
    @DisplayName("crear: OK")
    void crear_OK() {
        when(equipoRepository.existsByCodigoInventario("MIC-001")).thenReturn(false);
        when(equipoRepository.save(any(Equipo.class))).thenAnswer(inv -> {
            Equipo e = inv.getArgument(0);
            e.setId(1L);
            return e;
        });
        EquipoResponse r = equipoService.crear(validRequest);
        assertEquals(1L, r.getId());
        assertEquals("MIC-001", r.getCodigoInventario());
    }

    @Test
    @DisplayName("crear: codigo duplicado lanza DuplicateResourceException")
    void crear_Duplicado() {
        when(equipoRepository.existsByCodigoInventario("MIC-001")).thenReturn(true);
        assertThrows(DuplicateResourceException.class, () -> equipoService.crear(validRequest));
    }

    @Test
    @DisplayName("obtenerPorId: OK")
    void obtenerPorId_OK() {
        when(equipoRepository.findById(10L)).thenReturn(Optional.of(existente));
        EquipoResponse r = equipoService.obtenerPorId(10L);
        assertEquals("ESP-002", r.getCodigoInventario());
    }

    @Test
    @DisplayName("obtenerPorId: no existe -> ResourceNotFoundException")
    void obtenerPorId_NoExiste() {
        when(equipoRepository.findById(10L)).thenReturn(Optional.empty());
        assertThrows(ResourceNotFoundException.class, () -> equipoService.obtenerPorId(10L));
    }

    @Test
    @DisplayName("obtenerTodos: retorna mapeo")
    void obtenerTodos() {
        when(equipoRepository.findAll()).thenReturn(List.of(existente));
        assertEquals(1, equipoService.obtenerTodos().size());
    }

    @Test
    @DisplayName("actualizar: cambia el codigo si no esta tomado")
    void actualizar_OK() {
        validRequest.setCodigoInventario("NUEVO-XX");
        when(equipoRepository.findById(10L)).thenReturn(Optional.of(existente));
        when(equipoRepository.existsByCodigoInventario("NUEVO-XX")).thenReturn(false);
        when(equipoRepository.save(any(Equipo.class))).thenAnswer(inv -> inv.getArgument(0));

        EquipoResponse r = equipoService.actualizar(10L, validRequest);
        assertEquals("NUEVO-XX", r.getCodigoInventario());
        assertEquals(3, r.getCantidad());
    }

    @Test
    @DisplayName("actualizar: codigo cambia a uno tomado -> DuplicateResourceException")
    void actualizar_CodigoTomado() {
        validRequest.setCodigoInventario("OTRO-CODIGO");
        when(equipoRepository.findById(10L)).thenReturn(Optional.of(existente));
        when(equipoRepository.existsByCodigoInventario("OTRO-CODIGO")).thenReturn(true);
        assertThrows(DuplicateResourceException.class, () -> equipoService.actualizar(10L, validRequest));
    }

    @Test
    @DisplayName("actualizar: no existe -> ResourceNotFoundException")
    void actualizar_NoExiste() {
        when(equipoRepository.findById(10L)).thenReturn(Optional.empty());
        assertThrows(ResourceNotFoundException.class, () -> equipoService.actualizar(10L, validRequest));
    }

    @Test
    @DisplayName("eliminar: OK cuando no hay prestamos activos")
    void eliminar_OK() {
        when(equipoRepository.existsById(10L)).thenReturn(true);
        when(prestamoRepository.existsByEquipoIdAndEstado(10L, EstadoPrestamo.ACTIVO)).thenReturn(false);
        equipoService.eliminar(10L);
        verify(equipoRepository).deleteById(10L);
    }

    @Test
    @DisplayName("eliminar: no existe -> ResourceNotFoundException")
    void eliminar_NoExiste() {
        when(equipoRepository.existsById(10L)).thenReturn(false);
        assertThrows(ResourceNotFoundException.class, () -> equipoService.eliminar(10L));
        verify(equipoRepository, never()).deleteById(any());
    }

    @Test
    @DisplayName("eliminar: con prestamos activos -> BusinessRuleException")
    void eliminar_ConPrestamosActivos() {
        when(equipoRepository.existsById(10L)).thenReturn(true);
        when(prestamoRepository.existsByEquipoIdAndEstado(10L, EstadoPrestamo.ACTIVO)).thenReturn(true);
        assertThrows(BusinessRuleException.class, () -> equipoService.eliminar(10L));
        verify(equipoRepository, never()).deleteById(any());
    }
}
