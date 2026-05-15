package com.sgplab.backend.service.contract;

import com.sgplab.backend.dto.request.PenalizacionRequest;
import com.sgplab.backend.dto.response.PenalizacionResponse;

import java.util.List;

/**
 * Servicio para la gestion de penalizaciones.
 *
 * @author SGP LAB Team
 */
public interface IPenalizacionService {

    PenalizacionResponse crear(PenalizacionRequest request);

    PenalizacionResponse obtenerPorId(Long id);

    List<PenalizacionResponse> obtenerTodas();

    PenalizacionResponse actualizar(Long id, PenalizacionRequest request);

    void eliminar(Long id);

    boolean tienePenalizacionActiva(Long usuarioId);
}
