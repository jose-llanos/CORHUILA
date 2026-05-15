package com.sgplab.backend.service.impl;

import com.sgplab.backend.dto.request.LoginRequest;
import com.sgplab.backend.dto.response.LoginResponse;
import com.sgplab.backend.exception.InvalidCredentialsException;
import com.sgplab.backend.model.entity.Usuario;
import com.sgplab.backend.model.enums.EstadoUsuario;
import com.sgplab.backend.repository.IUsuarioRepository;
import com.sgplab.backend.security.JwtService;
import com.sgplab.backend.service.contract.IAuthService;
import com.sgplab.backend.util.PasswordHashUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/**
 * Implementacion del servicio de autenticacion.
 *
 * <p>Flujo de login:
 * <ol>
 *   <li>Busca al usuario por email.</li>
 *   <li>Verifica la contrasena contra el hash almacenado con {@link PasswordHashUtil}.</li>
 *   <li>Comprueba que el usuario este {@code ACTIVO}.</li>
 *   <li>Genera un JWT con {@link JwtService}.</li>
 * </ol>
 *
 * <p>Cualquier fallo en pasos 1-3 produce {@link InvalidCredentialsException}
 * con mensaje generico, para no facilitar enumeracion de cuentas.
 *
 * @author SGP LAB Team
 */
@Service
public class AuthServiceImpl implements IAuthService {

    private static final Logger log = LoggerFactory.getLogger(AuthServiceImpl.class);
    private static final String GENERIC_ERROR = "Credenciales invalidas.";

    private final IUsuarioRepository usuarioRepository;
    private final JwtService jwtService;

    public AuthServiceImpl(IUsuarioRepository usuarioRepository, JwtService jwtService) {
        this.usuarioRepository = usuarioRepository;
        this.jwtService = jwtService;
    }

    @Override
    public LoginResponse login(LoginRequest request) {
        Usuario usuario = usuarioRepository.findByEmail(request.getEmail())
                .orElseThrow(() -> {
                    log.info("Login fallido: email no registrado '{}'", request.getEmail());
                    return new InvalidCredentialsException(GENERIC_ERROR);
                });

        if (!PasswordHashUtil.matches(request.getPassword(), usuario.getPasswordHash())) {
            log.info("Login fallido: contrasena incorrecta para '{}'", request.getEmail());
            throw new InvalidCredentialsException(GENERIC_ERROR);
        }

        if (usuario.getEstado() != EstadoUsuario.ACTIVO) {
            log.info("Login fallido: usuario '{}' en estado {}", request.getEmail(), usuario.getEstado());
            throw new InvalidCredentialsException("La cuenta no esta activa.");
        }

        String token = jwtService.generateToken(usuario);
        log.info("Login exitoso para usuario id={} email={}", usuario.getId(), usuario.getEmail());
        return new LoginResponse(
                token,
                jwtService.getExpirationMs(),
                usuario.getId(),
                usuario.getEmail(),
                usuario.getNombre(),
                usuario.getRol()
        );
    }
}
