package com.sgplab.backend.service.impl;

import com.sgplab.backend.dto.mapper.PenalizacionMapper;
import com.sgplab.backend.dto.request.PenalizacionRequest;
import com.sgplab.backend.dto.response.PenalizacionResponse;
import com.sgplab.backend.exception.BusinessRuleException;
import com.sgplab.backend.exception.ResourceNotFoundException;
import com.sgplab.backend.model.entity.Penalizacion;
import com.sgplab.backend.model.entity.Usuario;
import com.sgplab.backend.model.enums.EstadoPenalizacion;
import com.sgplab.backend.repository.IPenalizacionRepository;
import com.sgplab.backend.repository.IUsuarioRepository;
import com.sgplab.backend.service.contract.IPenalizacionService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Implementacion del servicio de penalizaciones.
 *
 * <p>Reglas:
 * <ul>
 *   <li>{@code fechaFin} debe ser posterior o igual a {@code fechaInicio}.</li>
 *   <li>El usuario asociado debe existir.</li>
 * </ul>
 *
 * @author SGP LAB Team
 */
@Service
@Transactional
public class PenalizacionServiceImpl implements IPenalizacionService {

    private static final Logger log = LoggerFactory.getLogger(PenalizacionServiceImpl.class);
    private static final String RESOURCE = "Penalizacion";

    private final IPenalizacionRepository penalizacionRepository;
    private final IUsuarioRepository usuarioRepository;

    public PenalizacionServiceImpl(IPenalizacionRepository penalizacionRepository,
                                   IUsuarioRepository usuarioRepository) {
        this.penalizacionRepository = penalizacionRepository;
        this.usuarioRepository = usuarioRepository;
    }

    @Override
    public PenalizacionResponse crear(PenalizacionRequest request) {
        validarFechas(request);
        Usuario usuario = usuarioRepository.findById(request.getUsuarioId())
                .orElseThrow(() -> new ResourceNotFoundException("Usuario", "id", request.getUsuarioId()));

        Penalizacion p = new Penalizacion();
        p.setMotivo(request.getMotivo());
        p.setFechaInicio(request.getFechaInicio());
        p.setFechaFin(request.getFechaFin());
        p.setUsuario(usuario);
        p.setEstado(request.getEstado() != null ? request.getEstado() : EstadoPenalizacion.ACTIVA);

        Penalizacion guardada = penalizacionRepository.save(p);
        log.info("Penalizacion creada id={} usuarioId={}", guardada.getId(), usuario.getId());
        return PenalizacionMapper.toResponse(guardada);
    }

    @Override
    @Transactional(readOnly = true)
    public PenalizacionResponse obtenerPorId(Long id) {
        return PenalizacionMapper.toResponse(findOrThrow(id));
    }

    @Override
    @Transactional(readOnly = true)
    public List<PenalizacionResponse> obtenerTodas() {
        return penalizacionRepository.findAll().stream()
                .map(PenalizacionMapper::toResponse)
                .collect(Collectors.toList());
    }

    @Override
    public PenalizacionResponse actualizar(Long id, PenalizacionRequest request) {
        Penalizacion existente = findOrThrow(id);
        validarFechas(request);

        existente.setMotivo(request.getMotivo());
        existente.setFechaInicio(request.getFechaInicio());
        existente.setFechaFin(request.getFechaFin());
        if (request.getEstado() != null) {
            existente.setEstado(request.getEstado());
        }
        return PenalizacionMapper.toResponse(penalizacionRepository.save(existente));
    }

    @Override
    public void eliminar(Long id) {
        if (!penalizacionRepository.existsById(id)) {
            throw new ResourceNotFoundException(RESOURCE, "id", id);
        }
        penalizacionRepository.deleteById(id);
        log.info("Penalizacion eliminada id={}", id);
    }

    @Override
    @Transactional(readOnly = true)
    public boolean tienePenalizacionActiva(Long usuarioId) {
        return penalizacionRepository.existsByUsuarioIdAndEstado(usuarioId, EstadoPenalizacion.ACTIVA);
    }

    private void validarFechas(PenalizacionRequest request) {
        if (request.getFechaFin().isBefore(request.getFechaInicio())) {
            throw new BusinessRuleException("La fecha de fin no puede ser anterior a la fecha de inicio.");
        }
    }

    private Penalizacion findOrThrow(Long id) {
        return penalizacionRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(RESOURCE, "id", id));
    }
}
