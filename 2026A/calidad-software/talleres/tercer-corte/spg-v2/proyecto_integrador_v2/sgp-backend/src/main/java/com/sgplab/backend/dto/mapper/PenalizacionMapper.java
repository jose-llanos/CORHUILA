package com.sgplab.backend.dto.mapper;

import com.sgplab.backend.dto.response.PenalizacionResponse;
import com.sgplab.backend.model.entity.Penalizacion;

/**
 * Mapper de {@link Penalizacion} a su DTO de respuesta.
 *
 * @author SGP LAB Team
 */
public final class PenalizacionMapper {

    private PenalizacionMapper() {
        throw new UnsupportedOperationException("Clase de utilidad: no debe instanciarse.");
    }

    public static PenalizacionResponse toResponse(Penalizacion p) {
        if (p == null) {
            return null;
        }
        PenalizacionResponse r = new PenalizacionResponse();
        r.setId(p.getId());
        r.setMotivo(p.getMotivo());
        r.setFechaInicio(p.getFechaInicio());
        r.setFechaFin(p.getFechaFin());
        r.setEstado(p.getEstado());
        if (p.getUsuario() != null) {
            r.setUsuarioId(p.getUsuario().getId());
            r.setUsuarioNombre(p.getUsuario().getNombre());
        }
        return r;
    }
}
