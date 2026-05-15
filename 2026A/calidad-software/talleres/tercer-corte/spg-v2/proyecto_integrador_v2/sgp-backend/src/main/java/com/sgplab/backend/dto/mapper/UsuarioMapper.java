package com.sgplab.backend.dto.mapper;

import com.sgplab.backend.dto.request.UsuarioRequest;
import com.sgplab.backend.dto.response.UsuarioResponse;
import com.sgplab.backend.model.entity.Usuario;

/**
 * Mapper estatico entre la entidad {@link Usuario} y sus DTOs.
 * No expone nunca el {@code passwordHash} hacia el exterior.
 *
 * @author SGP LAB Team
 */
public final class UsuarioMapper {

    private UsuarioMapper() {
        throw new UnsupportedOperationException("Clase de utilidad: no debe instanciarse.");
    }

    /**
     * Convierte una entidad a su DTO de respuesta.
     *
     * @param usuario entidad a convertir; puede ser {@code null}.
     * @return DTO de respuesta o {@code null} si la entrada es nula.
     */
    public static UsuarioResponse toResponse(Usuario usuario) {
        if (usuario == null) {
            return null;
        }
        return new UsuarioResponse(
                usuario.getId(),
                usuario.getNombre(),
                usuario.getEmail(),
                usuario.getRol(),
                usuario.getEstado()
        );
    }

    /**
     * Construye una nueva entidad a partir del DTO de creacion.
     * No establece el {@code passwordHash}; eso es responsabilidad del servicio
     * (que aplica {@code PasswordHashUtil}).
     *
     * @param request DTO de entrada
     * @return entidad nueva (sin id ni passwordHash)
     */
    public static Usuario toEntity(UsuarioRequest request) {
        Usuario u = new Usuario();
        u.setNombre(request.getNombre());
        u.setEmail(request.getEmail());
        u.setRol(request.getRol());
        if (request.getEstado() != null) {
            u.setEstado(request.getEstado());
        }
        return u;
    }
}
