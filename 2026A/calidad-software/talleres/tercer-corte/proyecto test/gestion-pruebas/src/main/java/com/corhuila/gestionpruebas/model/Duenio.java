package com.corhuila.gestionpruebas.model;

import jakarta.persistence.*;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;

/**
 * Entidad que representa al dueño de la mascota.
 * Cumple con el requisito RF-01.
 */
@Entity
@Table(name = "duenios")
public class Duenio {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank(message = "El nombre es obligatorio") // RF-09
    private String nombre;

    @Email(message = "Ingrese un correo válido")
    @NotBlank(message = "El correo es obligatorio")
    private String correo;

    private String telefono;
    private String direccion;

    // Constructor vacío (Obligatorio para JPA)
    public Duenio() {}

    // Getters y Setters (Necesarios para que Thymeleaf vea los datos)
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getNombre() { return nombre; }
    public void setNombre(String nombre) { this.nombre = nombre; }
    public String getCorreo() { return correo; }
    public void setCorreo(String correo) { this.correo = correo; }
    public String getTelefono() { return telefono; }
    public void setTelefono(String telefono) { this.telefono = telefono; }
    public String getDireccion() { return direccion; }
    public void setDireccion(String direccion) { this.direccion = direccion; }
}