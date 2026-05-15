package com.sgplab.backend.controller;

import com.sgplab.backend.dto.request.LoginRequest;
import com.sgplab.backend.dto.response.LoginResponse;
import com.sgplab.backend.service.contract.IAuthService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Controlador REST para operaciones de autenticacion.
 *
 * @author SGP LAB Team
 */
@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final IAuthService authService;

    public AuthController(IAuthService authService) {
        this.authService = authService;
    }

    /**
     * Autentica un usuario y devuelve un JWT en caso de exito.
     *
     * @param request credenciales (email + password)
     * @return 200 OK con token y datos de usuario
     */
    @PostMapping("/login")
    public ResponseEntity<LoginResponse> login(@Valid @RequestBody LoginRequest request) {
        return ResponseEntity.ok(authService.login(request));
    }
}
