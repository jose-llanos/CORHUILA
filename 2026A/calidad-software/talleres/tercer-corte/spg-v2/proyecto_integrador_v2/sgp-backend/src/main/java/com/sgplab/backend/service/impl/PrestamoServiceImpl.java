package com.sgplab.backend.service.impl;

import com.sgplab.backend.dto.mapper.PrestamoMapper;
import com.sgplab.backend.dto.request.PrestamoRequest;
import com.sgplab.backend.dto.response.PrestamoResponse;
import com.sgplab.backend.exception.BusinessRuleException;
import com.sgplab.backend.exception.ResourceNotFoundException;
import com.sgplab.backend.model.entity.Equipo;
import com.sgplab.backend.model.entity.Prestamo;
import com.sgplab.backend.model.entity.Usuario;
import com.sgplab.backend.model.enums.EstadoPrestamo;
import com.sgplab.backend.repository.IEquipoRepository;
import com.sgplab.backend.repository.IPrestamoRepository;
import com.sgplab.backend.repository.IUsuarioRepository;
import com.sgplab.backend.service.contract.IPrestamoService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Implementacion del servicio de prestamos.
 *
 * <p>Reglas:
 * <ul>
 *   <li>Un usuario no puede tener mas de un prestamo en estado {@code ACTIVO}.</li>
 *   <li>El equipo debe existir y tener stock > 0.</li>
 *   <li>{@code fechaFin}, si se provee, debe ser posterior o igual a {@code fechaInicio}.</li>
 *   <li>Al crear un prestamo se descuenta 1 unidad del stock del equipo.</li>
 *   <li>Al pasar un prestamo de {@code ACTIVO} a {@code DEVUELTO} se devuelve 1 unidad al stock.</li>
 * </ul>
 *
 * @author SGP LAB Team
 */
@Service
@Transactional
public class PrestamoServiceImpl implements IPrestamoService {

    private static final Logger log = LoggerFactory.getLogger(PrestamoServiceImpl.class);
    private static final String RESOURCE = "Prestamo";

    private final IPrestamoRepository prestamoRepository;
    private final IEquipoRepository equipoRepository;
    private final IUsuarioRepository usuarioRepository;

    public PrestamoServiceImpl(IPrestamoRepository prestamoRepository,
                               IEquipoRepository equipoRepository,
                               IUsuarioRepository usuarioRepository) {
        this.prestamoRepository = prestamoRepository;
        this.equipoRepository = equipoRepository;
        this.usuarioRepository = usuarioRepository;
    }

    @Override
    public PrestamoResponse crear(PrestamoRequest request) {
        validarFechas(request);

        Usuario usuario = usuarioRepository.findById(request.getUsuarioId())
                .orElseThrow(() -> new ResourceNotFoundException("Usuario", "id", request.getUsuarioId()));

        Equipo equipo = equipoRepository.findById(request.getEquipoId())
                .orElseThrow(() -> new ResourceNotFoundException("Equipo", "id", request.getEquipoId()));

        if (prestamoRepository.existsByUsuarioIdAndEstado(usuario.getId(), EstadoPrestamo.ACTIVO)) {
            throw new BusinessRuleException(
                    "El usuario ya tiene un prestamo activo y debe devolverlo antes de solicitar otro.");
        }
        if (equipo.getCantidad() == null || equipo.getCantidad() <= 0) {
            throw new BusinessRuleException("No hay stock disponible para el equipo: " + equipo.getNombre());
        }

        equipo.setCantidad(equipo.getCantidad() - 1);
        equipoRepository.save(equipo);

        Prestamo prestamo = new Prestamo();
        prestamo.setUsuario(usuario);
        prestamo.setEquipo(equipo);
        prestamo.setFechaInicio(request.getFechaInicio());
        prestamo.setFechaFin(request.getFechaFin());
        prestamo.setEstado(EstadoPrestamo.ACTIVO);

        Prestamo guardado = prestamoRepository.save(prestamo);
        log.info("Prestamo creado id={} usuarioId={} equipoId={}",
                guardado.getId(), usuario.getId(), equipo.getId());
        return PrestamoMapper.toResponse(guardado);
    }

    @Override
    @Transactional(readOnly = true)
    public PrestamoResponse obtenerPorId(Long id) {
        return PrestamoMapper.toResponse(findOrThrow(id));
    }

    @Override
    @Transactional(readOnly = true)
    public List<PrestamoResponse> obtenerTodos() {
        return prestamoRepository.findAll().stream()
                .map(PrestamoMapper::toResponse)
                .collect(Collectors.toList());
    }

    @Override
    public PrestamoResponse actualizar(Long id, PrestamoRequest request) {
        Prestamo existente = findOrThrow(id);
        validarFechas(request);

        EstadoPrestamo nuevoEstado = request.getEstado() != null ? request.getEstado() : existente.getEstado();

        if (existente.getEstado() == EstadoPrestamo.ACTIVO && nuevoEstado == EstadoPrestamo.DEVUELTO) {
            Equipo equipo = existente.getEquipo();
            equipo.setCantidad(equipo.getCantidad() + 1);
            equipoRepository.save(equipo);
            log.info("Devolucion: stock equipo id={} restituido a {}", equipo.getId(), equipo.getCantidad());
        }

        existente.setFechaInicio(request.getFechaInicio());
        existente.setFechaFin(request.getFechaFin());
        existente.setEstado(nuevoEstado);

        return PrestamoMapper.toResponse(prestamoRepository.save(existente));
    }

    @Override
    public void eliminar(Long id) {
        Prestamo p = findOrThrow(id);
        if (p.getEstado() == EstadoPrestamo.ACTIVO) {
            throw new BusinessRuleException(
                    "No se puede eliminar un prestamo en estado ACTIVO. Primero registrelo como DEVUELTO o CANCELADO.");
        }
        prestamoRepository.deleteById(id);
        log.info("Prestamo eliminado id={}", id);
    }

    private void validarFechas(PrestamoRequest request) {
        if (request.getFechaFin() != null && request.getFechaFin().isBefore(request.getFechaInicio())) {
            throw new BusinessRuleException("La fecha de fin no puede ser anterior a la fecha de inicio.");
        }
    }

    private Prestamo findOrThrow(Long id) {
        return prestamoRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(RESOURCE, "id", id));
    }
}
