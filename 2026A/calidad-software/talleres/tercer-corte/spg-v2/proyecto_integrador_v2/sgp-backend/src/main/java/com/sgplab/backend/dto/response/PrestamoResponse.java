package com.sgplab.backend.dto.response;

import com.sgplab.backend.model.enums.EstadoPrestamo;

import java.time.LocalDate;

/**
 * DTO de salida para prestamos. Incluye referencias resumidas
 * al equipo y al usuario para evitar serializar la entidad completa.
 *
 * @author SGP LAB Team
 */
public class PrestamoResponse {

    private Long id;
    private LocalDate fechaInicio;
    private LocalDate fechaFin;
    private Long equipoId;
    private String equipoNombre;
    private Long usuarioId;
    private String usuarioNombre;
    private EstadoPrestamo estado;

    public PrestamoResponse() {
        // Para serializacion
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
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

    public Long getEquipoId() {
        return equipoId;
    }

    public void setEquipoId(Long equipoId) {
        this.equipoId = equipoId;
    }

    public String getEquipoNombre() {
        return equipoNombre;
    }

    public void setEquipoNombre(String equipoNombre) {
        this.equipoNombre = equipoNombre;
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

    public EstadoPrestamo getEstado() {
        return estado;
    }

    public void setEstado(EstadoPrestamo estado) {
        this.estado = estado;
    }
}
