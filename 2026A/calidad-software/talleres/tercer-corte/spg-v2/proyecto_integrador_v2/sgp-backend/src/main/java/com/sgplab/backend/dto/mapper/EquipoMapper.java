package com.sgplab.backend.dto.mapper;

import com.sgplab.backend.dto.request.EquipoRequest;
import com.sgplab.backend.dto.response.EquipoResponse;
import com.sgplab.backend.model.entity.Equipo;

/**
 * Mapper estatico entre la entidad {@link Equipo} y sus DTOs.
 *
 * @author SGP LAB Team
 */
public final class EquipoMapper {

    private EquipoMapper() {
        throw new UnsupportedOperationException("Clase de utilidad: no debe instanciarse.");
    }

    public static EquipoResponse toResponse(Equipo equipo) {
        if (equipo == null) {
            return null;
        }
        return new EquipoResponse(
                equipo.getId(),
                equipo.getCodigoInventario(),
                equipo.getNombre(),
                equipo.getCantidad(),
                equipo.getEstado()
        );
    }

    public static Equipo toEntity(EquipoRequest req) {
        Equipo e = new Equipo();
        e.setCodigoInventario(req.getCodigoInventario());
        e.setNombre(req.getNombre());
        e.setCantidad(req.getCantidad());
        if (req.getEstado() != null) {
            e.setEstado(req.getEstado());
        }
        return e;
    }
}
