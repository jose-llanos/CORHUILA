package com.map.parking_project.dto;

import java.time.LocalTime;

// Le dice a SonarQube que ignore todos los code smells de este archivo, pero sí medirá su cobertura.
@SuppressWarnings("all")
public class VehicleEntryDTO {
    
    private String placa;
    private String tipoVehiculo;
    private String ubicacion;
    private LocalTime horaIngreso; // Spring Boot convierte el string de Angular automáticamente

    // Getters y Setters
    public String getPlaca() {
        return placa;
    }

    public void setPlaca(String placa) {
        this.placa = placa;
    }

    public String getTipoVehiculo() {
        return tipoVehiculo;
    }

    public void setTipoVehiculo(String tipoVehiculo) {
        this.tipoVehiculo = tipoVehiculo;
    }

    public String getUbicacion() {
        return ubicacion;
    }

    public void setUbicacion(String ubicacion) {
        this.ubicacion = ubicacion;
    }

    public LocalTime getHoraIngreso() {
        return horaIngreso;
    }

    public void setHoraIngreso(LocalTime horaIngreso) {
        this.horaIngreso = horaIngreso;
    }
}