package com.sgplab.backend.service.impl;

import com.sgplab.backend.dto.mapper.EquipoMapper;
import com.sgplab.backend.dto.request.EquipoRequest;
import com.sgplab.backend.dto.response.EquipoResponse;
import com.sgplab.backend.exception.BusinessRuleException;
import com.sgplab.backend.exception.DuplicateResourceException;
import com.sgplab.backend.exception.ResourceNotFoundException;
import com.sgplab.backend.model.entity.Equipo;
import com.sgplab.backend.model.enums.EstadoPrestamo;
import com.sgplab.backend.repository.IEquipoRepository;
import com.sgplab.backend.repository.IPrestamoRepository;
import com.sgplab.backend.service.contract.IEquipoService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Implementacion del servicio de equipos.
 *
 * <p>Reglas:
 * <ul>
 *   <li>{@code codigoInventario} es unico (excepcion {@link DuplicateResourceException}).</li>
 *   <li>No se permite eliminar un equipo que aparezca en algun prestamo activo
 *       (excepcion {@link BusinessRuleException}).</li>
 * </ul>
 *
 * @author SGP LAB Team
 */
@Service
@Transactional
public class EquipoServiceImpl implements IEquipoService {

    private static final Logger log = LoggerFactory.getLogger(EquipoServiceImpl.class);
    private static final String RESOURCE = "Equipo";

    private final IEquipoRepository equipoRepository;
    private final IPrestamoRepository prestamoRepository;

    public EquipoServiceImpl(IEquipoRepository equipoRepository, IPrestamoRepository prestamoRepository) {
        this.equipoRepository = equipoRepository;
        this.prestamoRepository = prestamoRepository;
    }

    @Override
    public EquipoResponse crear(EquipoRequest request) {
        if (equipoRepository.existsByCodigoInventario(request.getCodigoInventario())) {
            throw new DuplicateResourceException(
                    "Ya existe un equipo con codigo de inventario: " + request.getCodigoInventario());
        }
        Equipo guardado = equipoRepository.save(EquipoMapper.toEntity(request));
        log.info("Equipo creado id={} codigo={}", guardado.getId(), guardado.getCodigoInventario());
        return EquipoMapper.toResponse(guardado);
    }

    @Override
    @Transactional(readOnly = true)
    public EquipoResponse obtenerPorId(Long id) {
        return EquipoMapper.toResponse(findOrThrow(id));
    }

    @Override
    @Transactional(readOnly = true)
    public List<EquipoResponse> obtenerTodos() {
        return equipoRepository.findAll().stream()
                .map(EquipoMapper::toResponse)
                .collect(Collectors.toList());
    }

    @Override
    public EquipoResponse actualizar(Long id, EquipoRequest request) {
        Equipo existente = findOrThrow(id);

        if (!existente.getCodigoInventario().equals(request.getCodigoInventario())
                && equipoRepository.existsByCodigoInventario(request.getCodigoInventario())) {
            throw new DuplicateResourceException(
                    "Ya existe otro equipo con codigo de inventario: " + request.getCodigoInventario());
        }

        existente.setNombre(request.getNombre());
        existente.setCodigoInventario(request.getCodigoInventario());
        existente.setCantidad(request.getCantidad());
        if (request.getEstado() != null) {
            existente.setEstado(request.getEstado());
        }
        return EquipoMapper.toResponse(equipoRepository.save(existente));
    }

    @Override
    public void eliminar(Long id) {
        if (!equipoRepository.existsById(id)) {
            throw new ResourceNotFoundException(RESOURCE, "id", id);
        }
        if (prestamoRepository.existsByEquipoIdAndEstado(id, EstadoPrestamo.ACTIVO)) {
            throw new BusinessRuleException(
                    "No se puede eliminar el equipo: tiene prestamos activos en curso.");
        }
        equipoRepository.deleteById(id);
        log.info("Equipo eliminado id={}", id);
    }

    private Equipo findOrThrow(Long id) {
        return equipoRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(RESOURCE, "id", id));
    }
}
