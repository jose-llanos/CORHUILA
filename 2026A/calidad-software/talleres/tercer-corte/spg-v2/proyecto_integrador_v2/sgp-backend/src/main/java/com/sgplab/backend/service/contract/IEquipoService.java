package com.sgplab.backend.service.contract;

import com.sgplab.backend.dto.request.EquipoRequest;
import com.sgplab.backend.dto.response.EquipoResponse;

import java.util.List;

/**
 * Servicio para la gestion de equipos de laboratorio.
 *
 * @author SGP LAB Team
 */
public interface IEquipoService {

    EquipoResponse crear(EquipoRequest request);

    EquipoResponse obtenerPorId(Long id);

    List<EquipoResponse> obtenerTodos();

    EquipoResponse actualizar(Long id, EquipoRequest request);

    void eliminar(Long id);
}
