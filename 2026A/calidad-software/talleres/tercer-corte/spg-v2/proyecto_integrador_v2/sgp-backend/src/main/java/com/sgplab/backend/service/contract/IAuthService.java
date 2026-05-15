package com.sgplab.backend.service.contract;

import com.sgplab.backend.dto.request.LoginRequest;
import com.sgplab.backend.dto.response.LoginResponse;

/**
 * Servicio de autenticacion.
 *
 * @author SGP LAB Team
 */
public interface IAuthService {

    /**
     * Autentica a un usuario con email y contrasena.
     *
     * @param request datos de login
     * @return respuesta con JWT y datos basicos del usuario
     * @throws com.sgplab.backend.exception.InvalidCredentialsException si las credenciales son incorrectas
     */
    LoginResponse login(LoginRequest request);
}
