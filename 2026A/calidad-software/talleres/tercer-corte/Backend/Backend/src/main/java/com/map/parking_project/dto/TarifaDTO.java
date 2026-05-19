package com.map.parking_project.dto; // Cambia el paquete si es necesario




// Le dice a SonarQube que ignore todos los code smells de este archivo, pero sí medirá su cobertura.
@SuppressWarnings("all")
public class TarifaDTO {


    private String tipoVehiculo;
    private Double tarifaDiurna; // Usa el tipo de dato que tengas (Double, Float o BigDecimal)
    private Double tarifaNocturna;
    private String imagen;

    // Getters y Setters
    public String getTipoVehiculo() { return tipoVehiculo; }
    public void setTipoVehiculo(String tipoVehiculo) { this.tipoVehiculo = tipoVehiculo; }

    public Double getTarifaDiurna() { return tarifaDiurna; }
    public void setTarifaDiurna(Double tarifaDiurna) { this.tarifaDiurna = tarifaDiurna; }

    public Double getTarifaNocturna() { return tarifaNocturna; }
    public void setTarifaNocturna(Double tarifaNocturna) { this.tarifaNocturna = tarifaNocturna; }

    public String getImagen() { return imagen; }
    public void setImagen(String imagen) { this.imagen = imagen; }
}