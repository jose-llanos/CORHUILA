package com.sgplab.backend.dto.response;

import com.sgplab.backend.model.enums.Rol;

/**
 * DTO de respuesta para un login exitoso.
 * Devuelve el JWT y los datos basicos del usuario autenticado.
 *
 * @author SGP LAB Team
 */
public class LoginResponse {

    private String token;
    private String tokenType = "Bearer";
    private long expiresInMs;
    private Long userId;
    private String email;
    private String nombre;
    private Rol rol;

    public LoginResponse() {
        // Para serializacion
    }

    public LoginResponse(String token, long expiresInMs, Long userId, String email, String nombre, Rol rol) {
        this.token = token;
        this.expiresInMs = expiresInMs;
        this.userId = userId;
        this.email = email;
        this.nombre = nombre;
        this.rol = rol;
    }

    public String getToken() {
        return token;
    }

    public void setToken(String token) {
        this.token = token;
    }

    public String getTokenType() {
        return tokenType;
    }

    public void setTokenType(String tokenType) {
        this.tokenType = tokenType;
    }

    public long getExpiresInMs() {
        return expiresInMs;
    }

    public void setExpiresInMs(long expiresInMs) {
        this.expiresInMs = expiresInMs;
    }

    public Long getUserId() {
        return userId;
    }

    public void setUserId(Long userId) {
        this.userId = userId;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getNombre() {
        return nombre;
    }

    public void setNombre(String nombre) {
        this.nombre = nombre;
    }

    public Rol getRol() {
        return rol;
    }

    public void setRol(Rol rol) {
        this.rol = rol;
    }
}
