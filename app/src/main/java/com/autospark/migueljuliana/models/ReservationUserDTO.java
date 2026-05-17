package com.autospark.migueljuliana.models;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ReservationUserDTO {

    private Long reservationId;
    private String customerFullName;
    private String customerIdentityCard;
    private String customerPhone;
    private String licensePlate;
    private VehicleType vehicleType;
    private String serviceType;
    private double value;
    private LocalDateTime reservationDate;
    private boolean isActive;
}