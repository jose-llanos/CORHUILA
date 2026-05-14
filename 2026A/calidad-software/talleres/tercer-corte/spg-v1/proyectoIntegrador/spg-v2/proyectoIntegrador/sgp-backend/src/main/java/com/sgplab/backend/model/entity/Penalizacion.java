package com.sgplab.backend.model.entity;

import com.sgplab.backend.model.enums.EstadoPenalizacion;
import jakarta.persistence.*;
import java.time.LocalDate;

@Entity
@Table(name = "penalizaciones")
public class Penalizacion {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String motivo;

    @Column(nullable = false)
    private LocalDate fechaInicio;

    @Column(nullable = false)
    private LocalDate fechaFin;

    // Relación con el Usuario que recibe la penalización
    @ManyToOne(optional = false)
    @JoinColumn(name = "usuario_id", nullable = false)
    private Usuario usuario;

    // NUEVO CAMPO: Estado de la penalización
    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private EstadoPenalizacion estado = EstadoPenalizacion.ACTIVA;

    public Penalizacion() {
        // Se añade este comentario para que SonarQube no lo marque como vacío
    }

    // Getters y Setters
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

    public Usuario getUsuario() {
        return usuario;
    }

    public void setUsuario(Usuario usuario) {
        this.usuario = usuario;
    }

    public EstadoPenalizacion getEstado() {
        return estado;
    }

    public void setEstado(EstadoPenalizacion estado) {
        this.estado = estado;
    }
}