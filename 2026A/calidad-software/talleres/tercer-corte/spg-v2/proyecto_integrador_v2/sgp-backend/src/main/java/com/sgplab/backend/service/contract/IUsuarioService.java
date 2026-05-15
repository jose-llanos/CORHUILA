package com.sgplab.backend.service.contract;

import com.sgplab.backend.dto.request.UsuarioRequest;
import com.sgplab.backend.dto.response.UsuarioResponse;

import java.util.List;

/**
 * Servicio para la gestion de usuarios.
 *
 * @author SGP LAB Team
 */
public interface IUsuarioService {

    UsuarioResponse crear(UsuarioRequest request);

    UsuarioResponse obtenerPorId(Long id);

    List<UsuarioResponse> obtenerTodos();

    UsuarioResponse actualizar(Long id, UsuarioRequest request);

    void eliminar(Long id);
}
