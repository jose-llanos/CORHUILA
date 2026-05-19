package com.map.parking_project.models;

import jakarta.persistence.*;

import java.io.Serializable;
import java.util.Date;

// Le dice a SonarQube que ignore todos los code smells de este archivo, pero sí medirá su cobertura.
@SuppressWarnings("all")
@Entity
@Table(name = "tarifa")
public class Tarifa implements Serializable {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "tipo_vehiculo", nullable = false)
    private String tipoVehiculo; // Ej: Automóvil, Moto, Camión, etc.

    @Column(name = "tarifa_diurna", nullable = false)
    private Double tarifaDiurna;

    @Column(name = "tarifa_nocturna", nullable = false)
    private Double tarifaNocturna;

    @Column(name = "imagen", nullable = false)
    private String imagen;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getTipoVehiculo() {
        return tipoVehiculo;
    }

    public void setTipoVehiculo(String tipoVehiculo) {
        this.tipoVehiculo = tipoVehiculo;
    }

    public Double getTarifaDiurna() {
        return tarifaDiurna;
    }

    public void setTarifaDiurna(Double tarifaDiurna) {
        this.tarifaDiurna = tarifaDiurna;
    }

    public Double getTarifaNocturna() {
        return tarifaNocturna;
    }

    public void setTarifaNocturna(Double tarifaNocturna) {
        this.tarifaNocturna = tarifaNocturna;
    }

    public String getImagen() {
        return imagen;
    }

    public void setImagen(String imagen) {
        this.imagen = imagen;
    }
}