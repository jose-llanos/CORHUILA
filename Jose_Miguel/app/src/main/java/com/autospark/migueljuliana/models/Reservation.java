package com.autospark.migueljuliana.models;

import jakarta.persistence.*;
import org.hibernate.envers.Audited;

import java.io.Serializable;
import java.time.LocalDateTime;

@Entity
@Audited
@Table(name = "reservas",
        uniqueConstraints = {
                @UniqueConstraint(name = "idx_unique_fecha_reserva",
                        columnNames = "fecha_reserva")
        })
public class Reservation implements Serializable {

    private static final long serialVersionUID = 1L;

    @Id
    @Column(nullable = false, name = "Id_Reserva")
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Enumerated(EnumType.STRING)
    @Column(name = "tipo_vehiculo", nullable = false)
    private VehicleType vehicleType;

    @Column(name = "licenseplate")
    private String licensePlate;

    @Column(name = "tipo_servicio", length = 100)
    private String serviceType;

    @Column(name = "valor")
    private Double value;

    @Column(name = "fecha_reserva", nullable = false)
    private LocalDateTime reservationDate;

    @Column(nullable = false)
    private boolean active;

    // Constructores
    public Reservation() {}

    public Reservation(Long id, VehicleType vehicleType, String licensePlate,
                       String serviceType, Double value,
                       LocalDateTime reservationDate, boolean active) {
        this.id = id;
        this.vehicleType = vehicleType;
        this.licensePlate = licensePlate;
        this.serviceType = serviceType;
        this.value = value;
        this.reservationDate = reservationDate;
        this.active = active;
    }

    // Getters y Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public VehicleType getVehicleType() { return vehicleType; }
    public void setVehicleType(VehicleType vehicleType) { this.vehicleType = vehicleType; }

    public String getLicensePlate() { return licensePlate; }
    public void setLicensePlate(String licensePlate) { this.licensePlate = licensePlate; }

    public String getServiceType() { return serviceType; }
    public void setServiceType(String serviceType) { this.serviceType = serviceType; }

    public Double getValue() { return value; }
    public void setValue(Double value) { this.value = value; }

    public LocalDateTime getReservationDate() { return reservationDate; }
    public void setReservationDate(LocalDateTime reservationDate) { this.reservationDate = reservationDate; }

    public boolean isActive() { return active; }
    public void setActive(boolean active) { this.active = active; }
}