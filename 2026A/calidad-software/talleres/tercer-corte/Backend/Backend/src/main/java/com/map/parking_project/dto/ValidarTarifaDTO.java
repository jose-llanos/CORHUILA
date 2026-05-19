package com.map.parking_project.dto;

// Le dice a SonarQube que ignore todos los code smells de este archivo, pero sí medirá su cobertura.
@SuppressWarnings("all")
public class ValidarTarifaDTO {
    private String plate;
    private String typecar;
    private double hours; // o int, dependiendo de cómo lo manejes en Angular

    public String getPlate() { return plate; }
    public void setPlate(String plate) { this.plate = plate; }

    public String getTypecar() { return typecar; }
    public void setTypecar(String typecar) { this.typecar = typecar; }

    public double getHours() { return hours; }
    public void setHours(double hours) { this.hours = hours; }
}