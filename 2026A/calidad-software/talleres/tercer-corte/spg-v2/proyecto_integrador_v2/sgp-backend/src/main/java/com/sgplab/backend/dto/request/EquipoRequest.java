package com.sgplab.backend.dto.request;

import com.sgplab.backend.model.enums.EstadoEquipo;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

/**
 * DTO de entrada para crear o actualizar un equipo.
 *
 * @author SGP LAB Team
 */
public class EquipoRequest {

    @NotBlank(message = "El codigo de inventario es obligatorio")
    @Size(min = 2, max = 50)
    private String codigoInventario;

    @NotBlank(message = "El nombre es obligatorio")
    @Size(min = 2, max = 150)
    private String nombre;

    @NotNull(message = "La cantidad es obligatoria")
    @Min(value = 0, message = "La cantidad no puede ser negativa")
    private Integer cantidad;

    private EstadoEquipo estado;

    public EquipoRequest() {
        // Para deserializacion
    }

    public String getCodigoInventario() {
        return codigoInventario;
    }

    public void setCodigoInventario(String codigoInventario) {
        this.codigoInventario = codigoInventario;
    }

    public String getNombre() {
        return nombre;
    }

    public void setNombre(String nombre) {
        this.nombre = nombre;
    }

    public Integer getCantidad() {
        return cantidad;
    }

    public void setCantidad(Integer cantidad) {
        this.cantidad = cantidad;
    }

    public EstadoEquipo getEstado() {
        return estado;
    }

    public void setEstado(EstadoEquipo estado) {
        this.estado = estado;
    }
}
