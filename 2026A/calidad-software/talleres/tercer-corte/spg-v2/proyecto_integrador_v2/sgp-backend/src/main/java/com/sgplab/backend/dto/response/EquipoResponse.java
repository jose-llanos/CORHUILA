package com.sgplab.backend.dto.response;

import com.sgplab.backend.model.enums.EstadoEquipo;

/**
 * DTO de salida para equipos.
 *
 * @author SGP LAB Team
 */
public class EquipoResponse {

    private Long id;
    private String codigoInventario;
    private String nombre;
    private Integer cantidad;
    private EstadoEquipo estado;

    public EquipoResponse() {
        // Para serializacion
    }

    public EquipoResponse(Long id, String codigoInventario, String nombre, Integer cantidad, EstadoEquipo estado) {
        this.id = id;
        this.codigoInventario = codigoInventario;
        this.nombre = nombre;
        this.cantidad = cantidad;
        this.estado = estado;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
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
