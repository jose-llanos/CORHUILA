package com.sgplab.backend.dto.response;

import com.sgplab.backend.model.enums.EstadoPenalizacion;

import java.time.LocalDate;

/**
 * DTO de salida para penalizaciones.
 *
 * @author SGP LAB Team
 */
public class PenalizacionResponse {

    private Long id;
    private String motivo;
    private LocalDate fechaInicio;
    private LocalDate fechaFin;
    private Long usuarioId;
    private String usuarioNombre;
    private EstadoPenalizacion estado;

    public PenalizacionResponse() {
        // Para serializacion
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getMotivo() {
        return motivo;
    }

    public void setMotivo(String motivo) {
        this.motivo = motivo;
    }

    public LocalDate getFechaInicio() {
        return fechaInicio;
    }

    public void setFechaInicio(LocalDate fechaInicio) {
        this.fechaInicio = fechaInicio;
    }

    public LocalDate getFechaFin() {
        return fechaFin;
    }

    public void setFechaFin(LocalDate fechaFin) {
        this.fechaFin = fechaFin;
    }

    public Long getUsuarioId() {
        return usuarioId;
    }

    public void setUsuarioId(Long usuarioId) {
        this.usuarioId = usuarioId;
    }

    public String getUsuarioNombre() {
        return usuarioNombre;
    }

    public void setUsuarioNombre(String usuarioNombre) {
        this.usuarioNombre = usuarioNombre;
    }

    public EstadoPenalizacion getEstado() {
        return estado;
    }

    public void setEstado(EstadoPenalizacion estado) {
        this.estado = estado;
    }
}
