package com.sgplab.backend.dto.request;

import com.sgplab.backend.model.enums.EstadoUsuario;
import com.sgplab.backend.model.enums.Rol;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

/**
 * DTO de entrada para la creacion y actualizacion de usuarios.
 * Para la creacion la contrasena es obligatoria; para la actualizacion
 * es opcional (si viene null, no se modifica).
 *
 * @author SGP LAB Team
 */
public class UsuarioRequest {

    @NotBlank(message = "El nombre es obligatorio")
    @Size(min = 2, max = 120, message = "El nombre debe tener entre 2 y 120 caracteres")
    private String nombre;

    @NotBlank(message = "El email es obligatorio")
    @Email(message = "El email debe tener formato valido")
    @Size(max = 150)
    private String email;

    /**
     * Contrasena en claro. Obligatoria al crear; opcional al actualizar.
     * Validada explicitamente en el servicio segun la operacion.
     */
    @Size(min = 6, max = 100, message = "La contrasena debe tener entre 6 y 100 caracteres")
    private String password;

    @NotNull(message = "El rol es obligatorio")
    private Rol rol;

    private EstadoUsuario estado;

    public UsuarioRequest() {
        // Para deserializacion
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

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
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
