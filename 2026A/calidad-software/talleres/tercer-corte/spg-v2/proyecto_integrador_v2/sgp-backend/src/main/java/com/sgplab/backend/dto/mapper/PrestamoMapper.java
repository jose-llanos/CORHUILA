package com.sgplab.backend.dto.mapper;

import com.sgplab.backend.dto.response.PrestamoResponse;
import com.sgplab.backend.model.entity.Prestamo;

/**
 * Mapper de {@link Prestamo} a su DTO de respuesta.
 *
 * @author SGP LAB Team
 */
public final class PrestamoMapper {

    private PrestamoMapper() {
        throw new UnsupportedOperationException("Clase de utilidad: no debe instanciarse.");
    }

    public static PrestamoResponse toResponse(Prestamo prestamo) {
        if (prestamo == null) {
            return null;
        }
        PrestamoResponse r = new PrestamoResponse();
        r.setId(prestamo.getId());
        r.setFechaInicio(prestamo.getFechaInicio());
        r.setFechaFin(prestamo.getFechaFin());
        r.setEstado(prestamo.getEstado());
        if (prestamo.getEquipo() != null) {
            r.setEquipoId(prestamo.getEquipo().getId());
            r.setEquipoNombre(prestamo.getEquipo().getNombre());
        }
        if (prestamo.getUsuario() != null) {
            r.setUsuarioId(prestamo.getUsuario().getId());
            r.setUsuarioNombre(prestamo.getUsuario().getNombre());
        }
        return r;
    }
}
