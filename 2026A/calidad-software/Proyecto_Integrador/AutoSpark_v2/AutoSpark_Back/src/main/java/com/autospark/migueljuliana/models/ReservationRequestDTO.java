package com.autospark.migueljuliana.models;

import java.time.LocalDate;
import java.time.LocalTime;

public class ReservationRequestDTO {
    private String vehicleType;
    private String licensePlate;
    private String serviceType;
    private Double value;
    private LocalDate reservationDate;
    private LocalTime reservationTime;
    private boolean active;

    // Constructor por defecto (NECESARIO)
    public ReservationRequestDTO() {}

    // Constructor con todos los campos
    public ReservationRequestDTO(String vehicleType, String licensePlate, String serviceType,
                                 Double value, LocalDate reservationDate, LocalTime reservationTime,
                                 boolean active) {
        this.vehicleType = vehicleType;
        this.licensePlate = licensePlate;
        this.serviceType = serviceType;
        this.value = value;
        this.reservationDate = reservationDate;
        this.reservationTime = reservationTime;
        this.active = active;
    }

    // Getters y Setters (NECESARIOS para que Spring convierta el JSON)
    public String getVehicleType() { return vehicleType; }
    public void setVehicleType(String vehicleType) { this.vehicleType = vehicleType; }

    public String getLicensePlate() { return licensePlate; }
    public void setLicensePlate(String licensePlate) { this.licensePlate = licensePlate; }

    public String getServiceType() { return serviceType; }
    public void setServiceType(String serviceType) { this.serviceType = serviceType; }

    public Double getValue() { return value; }
    public void setValue(Double value) { this.value = value; }

    public LocalDate getReservationDate() { return reservationDate; }
    public void setReservationDate(LocalDate reservationDate) { this.reservationDate = reservationDate; }

    public LocalTime getReservationTime() { return reservationTime; }
    public void setReservationTime(LocalTime reservationTime) { this.reservationTime = reservationTime; }

    public boolean isActive() { return active; }
    public void setActive(boolean active) { this.active = active; }
}