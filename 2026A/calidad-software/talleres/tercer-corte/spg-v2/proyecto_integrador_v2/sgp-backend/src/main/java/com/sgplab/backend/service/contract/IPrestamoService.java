package com.sgplab.backend.service.contract;

import com.sgplab.backend.dto.request.PrestamoRequest;
import com.sgplab.backend.dto.response.PrestamoResponse;

import java.util.List;

/**
 * Servicio para la gestion de prestamos.
 *
 * @author SGP LAB Team
 */
public interface IPrestamoService {

    PrestamoResponse crear(PrestamoRequest request);

    PrestamoResponse obtenerPorId(Long id);

    List<PrestamoResponse> obtenerTodos();

    PrestamoResponse actualizar(Long id, PrestamoRequest request);

    void eliminar(Long id);
}
