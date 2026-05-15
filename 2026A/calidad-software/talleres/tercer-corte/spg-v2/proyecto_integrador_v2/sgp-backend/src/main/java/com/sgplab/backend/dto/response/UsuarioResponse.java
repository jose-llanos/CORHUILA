package com.sgplab.backend.dto.response;

import com.sgplab.backend.model.enums.EstadoUsuario;
import com.sgplab.backend.model.enums.Rol;

/**
 * DTO de salida para usuarios. Nunca expone el hash de la contrasena.
 *
 * @author SGP LAB Team
 */
public class UsuarioResponse {

    private Long id;
    private String nombre;
    private String email;
    private Rol rol;
    private EstadoUsuario estado;

    public UsuarioResponse() {
        // Para serializacion
    }

    public UsuarioResponse(Long id, String nombre, String email, Rol rol, EstadoUsuario estado) {
        this.id = id;
        this.nombre = nombre;
        this.email = email;
        this.rol = rol;
        this.estado = estado;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getNombre() {
        return nombre;
    }

    public void setNombre(String nombre) {
        this.nombre = nombre;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public Rol getRol() {
        return rol;
    }

    public void setRol(Rol rol) {
        this.rol = rol;
    }

    public EstadoUsuario getEstado() {
        return estado;
    }

    public void setEstado(EstadoUsuario estado) {
        this.estado = estado;
    }
}
