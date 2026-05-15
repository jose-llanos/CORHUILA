package com.sgplab.backend.dto.request;

import com.sgplab.backend.model.enums.EstadoPenalizacion;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

import java.time.LocalDate;

/**
 * DTO de entrada para crear o actualizar una penalizacion.
 *
 * @author SGP LAB Team
 */
public class PenalizacionRequest {

    @NotBlank(message = "El motivo es obligatorio")
    @Size(min = 5, max = 250)
    private String motivo;

    @NotNull(message = "La fecha de inicio es obligatoria")
    private LocalDate fechaInicio;

    @NotNull(message = "La fecha de fin es obligatoria")
    private LocalDate fechaFin;

    @NotNull(message = "El id del usuario es obligatorio")
    private Long usuarioId;

    private EstadoPenalizacion estado;

    public PenalizacionRequest() {
        // Para deserializacion
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

    public EstadoPenalizacion getEstado() {
        return estado;
    }

    public void setEstado(EstadoPenalizacion estado) {
        this.estado = estado;
    }
}
