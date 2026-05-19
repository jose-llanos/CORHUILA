package com.autospark.migueljuliana.models;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.envers.Audited;

import java.io.Serializable;
import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Audited
@Table(name = "ventas")
public class Sale implements Serializable {

    private static final long serialVersionUID = 1L;

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String customerName;

    @Column(nullable = false)
    private String customerIdentification;

    @Column(nullable = false)
    private String customerPhone;

    @Column(nullable = false)
    private String vehiclePlate;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private VehicleType vehicleType;

    @Column(nullable = false)
    private String serviceType;

    @Column(nullable = false)
    private Double amount;

    @Column(nullable = false)
    private LocalDateTime saleDate;

    @Column(nullable = false)
    private boolean active;
}