package com.sgplab.backend.unitarias.service;

import com.sgplab.backend.dto.request.PrestamoRequest;
import com.sgplab.backend.dto.response.PrestamoResponse;
import com.sgplab.backend.exception.BusinessRuleException;
import com.sgplab.backend.exception.ResourceNotFoundException;
import com.sgplab.backend.model.entity.Equipo;
import com.sgplab.backend.model.entity.Prestamo;
import com.sgplab.backend.model.entity.Usuario;
import com.sgplab.backend.model.enums.EstadoEquipo;
import com.sgplab.backend.model.enums.EstadoPrestamo;
import com.sgplab.backend.repository.IEquipoRepository;
import com.sgplab.backend.repository.IPrestamoRepository;
import com.sgplab.backend.repository.IUsuarioRepository;
import com.sgplab.backend.service.impl.PrestamoServiceImpl;
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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class PrestamoServiceImplTest {

    @Mock private IPrestamoRepository prestamoRepository;
    @Mock private IEquipoRepository equipoRepository;
    @Mock private IUsuarioRepository usuarioRepository;
    @InjectMocks private PrestamoServiceImpl prestamoService;

    private Usuario usuario;
    private Equipo equipo;
    private PrestamoRequest request;

    @BeforeEach
    void setUp() {
        usuario = new Usuario();
        usuario.setId(1L);
        usuario.setNombre("Cliente");
        usuario.setEmail("c@x.com");

        equipo = new Equipo();
        equipo.setId(2L);
        equipo.setNombre("Microscopio");
        equipo.setCantidad(5);
        equipo.setEstado(EstadoEquipo.DISPONIBLE);

        request = new PrestamoRequest();
        request.setUsuarioId(1L);
        request.setEquipoId(2L);
        request.setFechaInicio(LocalDate.now());
        request.setFechaFin(LocalDate.now().plusDays(3));
    }

    // -------- crear --------

    @Test
    @DisplayName("crear: flujo normal descuenta stock y asigna estado ACTIVO")
    void crear_OK() {
        when(usuarioRepository.findById(1L)).thenReturn(Optional.of(usuario));
        when(equipoRepository.findById(2L)).thenReturn(Optional.of(equipo));
        when(prestamoRepository.existsByUsuarioIdAndEstado(1L, EstadoPrestamo.ACTIVO)).thenReturn(false);
        when(prestamoRepository.save(any(Prestamo.class))).thenAnswer(inv -> {
            Prestamo p = inv.getArgument(0);
            p.setId(100L);
            return p;
        });

        PrestamoResponse r = prestamoService.crear(request);

        assertNotNull(r);
        assertEquals(100L, r.getId());
        assertEquals(EstadoPrestamo.ACTIVO, r.getEstado());
        assertEquals(4, equipo.getCantidad(), "Stock debio decrementarse");
        verify(equipoRepository).save(equipo);
    }

    @Test
    @DisplayName("crear: usuario inexistente -> ResourceNotFoundException")
    void crear_UsuarioNoExiste() {
        when(usuarioRepository.findById(1L)).thenReturn(Optional.empty());
        assertThrows(ResourceNotFoundException.class, () -> prestamoService.crear(request));
    }

    @Test
    @DisplayName("crear: equipo inexistente -> ResourceNotFoundException")
    void crear_EquipoNoExiste() {
        when(usuarioRepository.findById(1L)).thenReturn(Optional.of(usuario));
        when(equipoRepository.findById(2L)).thenReturn(Optional.empty());
        assertThrows(ResourceNotFoundException.class, () -> prestamoService.crear(request));
    }

    @Test
    @DisplayName("crear: usuario con prestamo activo -> BusinessRuleException")
    void crear_UsuarioConPrestamoActivo() {
        when(usuarioRepository.findById(1L)).thenReturn(Optional.of(usuario));
        when(equipoRepository.findById(2L)).thenReturn(Optional.of(equipo));
        when(prestamoRepository.existsByUsuarioIdAndEstado(1L, EstadoPrestamo.ACTIVO)).thenReturn(true);
        assertThrows(BusinessRuleException.class, () -> prestamoService.crear(request));
        verify(prestamoRepository, never()).save(any());
    }

    @Test
    @DisplayName("crear: stock 0 -> BusinessRuleException")
    void crear_SinStock() {
        equipo.setCantidad(0);
        when(usuarioRepository.findById(1L)).thenReturn(Optional.of(usuario));
        when(equipoRepository.findById(2L)).thenReturn(Optional.of(equipo));
        when(prestamoRepository.existsByUsuarioIdAndEstado(1L, EstadoPrestamo.ACTIVO)).thenReturn(false);
        assertThrows(BusinessRuleException.class, () -> prestamoService.crear(request));
    }

    @Test
    @DisplayName("crear: fechaFin antes de fechaInicio -> BusinessRuleException")
    void crear_FechasInvalidas() {
        request.setFechaFin(LocalDate.now().minusDays(1));
        assertThrows(BusinessRuleException.class, () -> prestamoService.crear(request));
    }

    // -------- obtener --------

    @Test
    @DisplayName("obtenerPorId: OK")
    void obtenerPorId_OK() {
        Prestamo p = new Prestamo();
        p.setId(5L);
        p.setUsuario(usuario);
        p.setEquipo(equipo);
        p.setEstado(EstadoPrestamo.ACTIVO);
        when(prestamoRepository.findById(5L)).thenReturn(Optional.of(p));
        assertEquals(5L, prestamoService.obtenerPorId(5L).getId());
    }

    @Test
    @DisplayName("obtenerPorId: no existe -> ResourceNotFoundException")
    void obtenerPorId_NoExiste() {
        when(prestamoRepository.findById(5L)).thenReturn(Optional.empty());
        assertThrows(ResourceNotFoundException.class, () -> prestamoService.obtenerPorId(5L));
    }

    @Test
    @DisplayName("obtenerTodos: OK")
    void obtenerTodos() {
        Prestamo p = new Prestamo();
        p.setId(1L);
        p.setUsuario(usuario);
        p.setEquipo(equipo);
        when(prestamoRepository.findAll()).thenReturn(List.of(p));
        assertEquals(1, prestamoService.obtenerTodos().size());
    }

    // -------- actualizar --------

    @Test
    @DisplayName("actualizar: ACTIVO -> DEVUELTO devuelve stock")
    void actualizar_Devolucion() {
        Prestamo existente = new Prestamo();
        existente.setId(5L);
        existente.setUsuario(usuario);
        existente.setEquipo(equipo);
        existente.setEstado(EstadoPrestamo.ACTIVO);
        equipo.setCantidad(3);

        when(prestamoRepository.findById(5L)).thenReturn(Optional.of(existente));
        when(prestamoRepository.save(any(Prestamo.class))).thenAnswer(inv -> inv.getArgument(0));

        request.setEstado(EstadoPrestamo.DEVUELTO);
        prestamoService.actualizar(5L, request);

        assertEquals(4, equipo.getCantidad(), "Stock debe restituirse en 1");
        verify(equipoRepository).save(equipo);
    }

    @Test
    @DisplayName("actualizar: cambio de estado distinto a devolucion no toca stock")
    void actualizar_SinDevolucion() {
        Prestamo existente = new Prestamo();
        existente.setId(5L);
        existente.setUsuario(usuario);
        existente.setEquipo(equipo);
        existente.setEstado(EstadoPrestamo.ACTIVO);

        when(prestamoRepository.findById(5L)).thenReturn(Optional.of(existente));
        when(prestamoRepository.save(any(Prestamo.class))).thenAnswer(inv -> inv.getArgument(0));

        request.setEstado(EstadoPrestamo.VENCIDO);
        prestamoService.actualizar(5L, request);

        verify(equipoRepository, never()).save(any());
    }

    @Test
    @DisplayName("actualizar: no existe -> ResourceNotFoundException")
    void actualizar_NoExiste() {
        when(prestamoRepository.findById(5L)).thenReturn(Optional.empty());
        assertThrows(ResourceNotFoundException.class, () -> prestamoService.actualizar(5L, request));
    }

    // -------- eliminar --------

    @Test
    @DisplayName("eliminar: prestamo ACTIVO -> BusinessRuleException")
    void eliminar_Activo() {
        Prestamo p = new Prestamo();
        p.setId(5L);
        p.setEstado(EstadoPrestamo.ACTIVO);
        p.setUsuario(usuario);
        p.setEquipo(equipo);
        when(prestamoRepository.findById(5L)).thenReturn(Optional.of(p));
        assertThrows(BusinessRuleException.class, () -> prestamoService.eliminar(5L));
        verify(prestamoRepository, never()).deleteById(any());
    }

    @Test
    @DisplayName("eliminar: prestamo DEVUELTO -> OK")
    void eliminar_Devuelto() {
        Prestamo p = new Prestamo();
        p.setId(5L);
        p.setEstado(EstadoPrestamo.DEVUELTO);
        p.setUsuario(usuario);
        p.setEquipo(equipo);
        when(prestamoRepository.findById(5L)).thenReturn(Optional.of(p));
        prestamoService.eliminar(5L);
        verify(prestamoRepository).deleteById(5L);
    }

    @Test
    @DisplayName("eliminar: no existe -> ResourceNotFoundException")
    void eliminar_NoExiste() {
        when(prestamoRepository.findById(5L)).thenReturn(Optional.empty());
        assertThrows(ResourceNotFoundException.class, () -> prestamoService.eliminar(5L));
    }
}
