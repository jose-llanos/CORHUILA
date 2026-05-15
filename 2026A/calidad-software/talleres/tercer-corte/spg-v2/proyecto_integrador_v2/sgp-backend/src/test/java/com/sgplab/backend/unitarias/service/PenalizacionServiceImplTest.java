package com.sgplab.backend.unitarias.service;

import com.sgplab.backend.dto.request.PenalizacionRequest;
import com.sgplab.backend.dto.response.PenalizacionResponse;
import com.sgplab.backend.exception.BusinessRuleException;
import com.sgplab.backend.exception.ResourceNotFoundException;
import com.sgplab.backend.model.entity.Penalizacion;
import com.sgplab.backend.model.entity.Usuario;
import com.sgplab.backend.model.enums.EstadoPenalizacion;
import com.sgplab.backend.repository.IPenalizacionRepository;
import com.sgplab.backend.repository.IUsuarioRepository;
import com.sgplab.backend.service.impl.PenalizacionServiceImpl;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class PenalizacionServiceImplTest {

    @Mock private IPenalizacionRepository penalizacionRepository;
    @Mock private IUsuarioRepository usuarioRepository;
    @InjectMocks private PenalizacionServiceImpl penalizacionService;

    private Usuario usuario;
    private PenalizacionRequest request;
    private Penalizacion existente;

    @BeforeEach
    void setUp() {
        usuario = new Usuario();
        usuario.setId(1L);
        usuario.setNombre("Cliente X");

        request = new PenalizacionRequest();
        request.setMotivo("Devolucion tardia");
        request.setFechaInicio(LocalDate.now());
        request.setFechaFin(LocalDate.now().plusDays(7));
        request.setUsuarioId(1L);

        existente = new Penalizacion();
        existente.setId(50L);
        existente.setUsuario(usuario);
        existente.setMotivo("Antiguo");
        existente.setFechaInicio(LocalDate.now().minusDays(2));
        existente.setFechaFin(LocalDate.now().plusDays(2));
        existente.setEstado(EstadoPenalizacion.ACTIVA);
    }

    @Test
    @DisplayName("crear: OK")
    void crear_OK() {
        when(usuarioRepository.findById(1L)).thenReturn(Optional.of(usuario));
        when(penalizacionRepository.save(any(Penalizacion.class))).thenAnswer(inv -> {
            Penalizacion p = inv.getArgument(0);
            p.setId(99L);
            return p;
        });
        PenalizacionResponse r = penalizacionService.crear(request);
        assertEquals(99L, r.getId());
        assertEquals(EstadoPenalizacion.ACTIVA, r.getEstado());
        assertEquals(1L, r.getUsuarioId());
    }

    @Test
    @DisplayName("crear: usuario no existe -> ResourceNotFoundException")
    void crear_UsuarioNoExiste() {
        when(usuarioRepository.findById(1L)).thenReturn(Optional.empty());
        assertThrows(ResourceNotFoundException.class, () -> penalizacionService.crear(request));
    }

    @Test
    @DisplayName("crear: fechaFin antes de fechaInicio -> BusinessRuleException")
    void crear_FechasInvalidas() {
        request.setFechaInicio(LocalDate.now());
        request.setFechaFin(LocalDate.now().minusDays(1));
        assertThrows(BusinessRuleException.class, () -> penalizacionService.crear(request));
        verify(penalizacionRepository, never()).save(any());
    }

    @Test
    @DisplayName("obtenerPorId: OK")
    void obtenerPorId_OK() {
        when(penalizacionRepository.findById(50L)).thenReturn(Optional.of(existente));
        assertEquals(50L, penalizacionService.obtenerPorId(50L).getId());
    }

    @Test
    @DisplayName("obtenerPorId: no existe")
    void obtenerPorId_NoExiste() {
        when(penalizacionRepository.findById(50L)).thenReturn(Optional.empty());
        assertThrows(ResourceNotFoundException.class, () -> penalizacionService.obtenerPorId(50L));
    }

    @Test
    @DisplayName("obtenerTodas")
    void obtenerTodas() {
        when(penalizacionRepository.findAll()).thenReturn(List.of(existente));
        assertEquals(1, penalizacionService.obtenerTodas().size());
    }

    @Test
    @DisplayName("actualizar: OK")
    void actualizar_OK() {
        when(penalizacionRepository.findById(50L)).thenReturn(Optional.of(existente));
        when(penalizacionRepository.save(any(Penalizacion.class))).thenAnswer(inv -> inv.getArgument(0));

        request.setMotivo("Nuevo motivo");
        request.setEstado(EstadoPenalizacion.LEVANTADA);

        PenalizacionResponse r = penalizacionService.actualizar(50L, request);
        assertEquals("Nuevo motivo", r.getMotivo());
        assertEquals(EstadoPenalizacion.LEVANTADA, r.getEstado());
    }

    @Test
    @DisplayName("actualizar: no existe -> ResourceNotFoundException")
    void actualizar_NoExiste() {
        when(penalizacionRepository.findById(50L)).thenReturn(Optional.empty());
        assertThrows(ResourceNotFoundException.class, () -> penalizacionService.actualizar(50L, request));
    }

    @Test
    @DisplayName("eliminar: OK")
    void eliminar_OK() {
        when(penalizacionRepository.existsById(50L)).thenReturn(true);
        penalizacionService.eliminar(50L);
        verify(penalizacionRepository).deleteById(50L);
    }

    @Test
    @DisplayName("eliminar: no existe -> ResourceNotFoundException")
    void eliminar_NoExiste() {
        when(penalizacionRepository.existsById(50L)).thenReturn(false);
        assertThrows(ResourceNotFoundException.class, () -> penalizacionService.eliminar(50L));
    }

    @Test
    @DisplayName("tienePenalizacionActiva: true cuando hay activa")
    void tienePenalizacion_True() {
        when(penalizacionRepository.existsByUsuarioIdAndEstado(1L, EstadoPenalizacion.ACTIVA)).thenReturn(true);
        assertTrue(penalizacionService.tienePenalizacionActiva(1L));
    }

    @Test
    @DisplayName("tienePenalizacionActiva: false cuando no hay")
    void tienePenalizacion_False() {
        when(penalizacionRepository.existsByUsuarioIdAndEstado(1L, EstadoPenalizacion.ACTIVA)).thenReturn(false);
        assertFalse(penalizacionService.tienePenalizacionActiva(1L));
    }
}
