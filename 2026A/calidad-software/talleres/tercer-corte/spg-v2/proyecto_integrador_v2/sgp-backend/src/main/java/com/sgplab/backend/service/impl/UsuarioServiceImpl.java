package com.sgplab.backend.service.impl;

import com.sgplab.backend.dto.mapper.UsuarioMapper;
import com.sgplab.backend.dto.request.UsuarioRequest;
import com.sgplab.backend.dto.response.UsuarioResponse;
import com.sgplab.backend.exception.BusinessRuleException;
import com.sgplab.backend.exception.DuplicateResourceException;
import com.sgplab.backend.exception.ResourceNotFoundException;
import com.sgplab.backend.model.entity.Usuario;
import com.sgplab.backend.repository.IUsuarioRepository;
import com.sgplab.backend.service.contract.IUsuarioService;
import com.sgplab.backend.util.PasswordHashUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Implementacion del servicio de usuarios.
 *
 * <p>Reglas:
 * <ul>
 *   <li>Email unico (excepcion {@link DuplicateResourceException}).</li>
 *   <li>Al crear, la contrasena es obligatoria y se almacena hasheada.</li>
 *   <li>Al actualizar, la contrasena es opcional: si viene nula o vacia, se conserva la actual.</li>
 * </ul>
 *
 * @author SGP LAB Team
 */
@Service
@Transactional
public class UsuarioServiceImpl implements IUsuarioService {

    private static final Logger log = LoggerFactory.getLogger(UsuarioServiceImpl.class);
    private static final String RESOURCE = "Usuario";

    private final IUsuarioRepository usuarioRepository;

    public UsuarioServiceImpl(IUsuarioRepository usuarioRepository) {
        this.usuarioRepository = usuarioRepository;
    }

    @Override
    public UsuarioResponse crear(UsuarioRequest request) {
        if (request.getPassword() == null || request.getPassword().isBlank()) {
            throw new BusinessRuleException("La contrasena es obligatoria al crear un usuario.");
        }
        if (usuarioRepository.existsByEmail(request.getEmail())) {
            throw new DuplicateResourceException("Ya existe un usuario con el email: " + request.getEmail());
        }
        Usuario usuario = UsuarioMapper.toEntity(request);
        usuario.setPasswordHash(PasswordHashUtil.hash(request.getPassword()));
        Usuario guardado = usuarioRepository.save(usuario);
        log.info("Usuario creado id={} email={} rol={}", guardado.getId(), guardado.getEmail(), guardado.getRol());
        return UsuarioMapper.toResponse(guardado);
    }

    @Override
    @Transactional(readOnly = true)
    public UsuarioResponse obtenerPorId(Long id) {
        Usuario usuario = findOrThrow(id);
        return UsuarioMapper.toResponse(usuario);
    }

    @Override
    @Transactional(readOnly = true)
    public List<UsuarioResponse> obtenerTodos() {
        return usuarioRepository.findAll().stream()
                .map(UsuarioMapper::toResponse)
                .collect(Collectors.toList());
    }

    @Override
    public UsuarioResponse actualizar(Long id, UsuarioRequest request) {
        Usuario existente = findOrThrow(id);

        if (!existente.getEmail().equals(request.getEmail())
                && usuarioRepository.existsByEmail(request.getEmail())) {
            throw new DuplicateResourceException("Ya existe otro usuario con el email: " + request.getEmail());
        }

        existente.setNombre(request.getNombre());
        existente.setEmail(request.getEmail());
        existente.setRol(request.getRol());
        if (request.getEstado() != null) {
            existente.setEstado(request.getEstado());
        }
        if (request.getPassword() != null && !request.getPassword().isBlank()) {
            existente.setPasswordHash(PasswordHashUtil.hash(request.getPassword()));
            log.info("Contrasena actualizada para usuario id={}", id);
        }
        Usuario actualizado = usuarioRepository.save(existente);
        return UsuarioMapper.toResponse(actualizado);
    }

    @Override
    public void eliminar(Long id) {
        if (!usuarioRepository.existsById(id)) {
            throw new ResourceNotFoundException(RESOURCE, "id", id);
        }
        usuarioRepository.deleteById(id);
        log.info("Usuario eliminado id={}", id);
    }

    private Usuario findOrThrow(Long id) {
        return usuarioRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(RESOURCE, "id", id));
    }
}
