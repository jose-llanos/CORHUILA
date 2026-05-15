package com.autospark.migueljuliana.controllers;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.autospark.migueljuliana.services.IReservationService;
import com.autospark.migueljuliana.models.Reservation;
import com.autospark.migueljuliana.models.ReservationRequestDTO;
import com.autospark.migueljuliana.models.ReservationUserDTO;
import com.autospark.migueljuliana.models.VehicleType;
import com.autospark.migueljuliana.exception.ResourceNotFoundException;

import java.time.format.DateTimeFormatter;
import java.util.Collections;
import java.util.Optional;

import java.time.LocalDateTime;
import java.util.List;

@Slf4j
@CrossOrigin(origins = "http://localhost:4200", allowCredentials = "true")
@RestController
@RequestMapping("/autospark")
@RequiredArgsConstructor
public class ReservationController {

    private static final String RESERVATION_ENTITY = "Reservation";

    private final IReservationService reservationService;

    @GetMapping("/reserva")
    public ResponseEntity<List<Reservation>> getAllReservations() {
        log.debug("Obteniendo todas las reservas");
        List<Reservation> reservations = reservationService.findAll();
        return ResponseEntity.ok(reservations);
    }

    @GetMapping("/reserva/{id}")
    public ResponseEntity<Reservation> getReservationById(@PathVariable Long id) {
        log.debug("Buscando reserva con id: {}", id);

        Reservation reservation = reservationService.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(RESERVATION_ENTITY, id));

        return ResponseEntity.ok(reservation);
    }

    @PostMapping("/reserva")
    public ResponseEntity<Object> createReservation(@RequestBody ReservationRequestDTO request) {
        log.info("Creando nueva reserva - Vehículo: {}, Servicio: {}, Placa: {}, Valor: {}, Fecha: {}, Hora: {}",
                request.getVehicleType(),
                request.getServiceType(),
                request.getLicensePlate(),
                request.getValue(),
                request.getReservationDate(),
                request.getReservationTime());

        // ERROR MEDIUM 1: Potencial NullPointerException - No validar que request no sea null
        // Si el cliente envía un body vacío o mal formado, request puede ser null
        if (request == null) {
            // Esta validación debería estar al principio
            return ResponseEntity.badRequest().body("Request body cannot be null");
        }

        // ERROR MEDIUM 2: Múltiples potenciales NullPointerException
        // No se validan los campos obligatorios antes de usarlos
        if (request.getVehicleType() == null || request.getVehicleType().trim().isEmpty()) {
            log.error("Vehicle type is null or empty");
            return ResponseEntity.badRequest().body("Vehicle type is required");
        }

        if (request.getServiceType() == null || request.getServiceType().trim().isEmpty()) {
            log.error("Service type is null or empty");
            return ResponseEntity.badRequest().body("Service type is required");
        }

        if (request.getLicensePlate() == null || request.getLicensePlate().trim().isEmpty()) {
            log.error("License plate is null or empty");
            return ResponseEntity.badRequest().body("License plate is required");
        }

        if (request.getReservationDate() == null) {
            log.error("Reservation date is null");
            return ResponseEntity.badRequest().body("Reservation date is required");
        }

        if (request.getReservationTime() == null) {
            log.error("Reservation time is null");
            return ResponseEntity.badRequest().body("Reservation time is required");
        }

        // ERROR MEDIUM 3: Posible IllegalArgumentException si el valor del enum no es válido
        // VehicleType.valueOf() puede lanzar IllegalArgumentException si el string no coincide
        VehicleType vehicleTypeEnum;
        try {
            vehicleTypeEnum = VehicleType.valueOf(request.getVehicleType());
        } catch (IllegalArgumentException e) {
            log.error("Invalid vehicle type: {}", request.getVehicleType());
            return ResponseEntity.badRequest().body("Invalid vehicle type. Allowed values: CAR, MOTORCYCLE, TRUCK, SUV");
        }

        // ERROR MEDIUM 4: No validación de formato de placa
        // La placa debería tener un formato específico (ej: ABC123)
        if (!request.getLicensePlate().matches("^[A-Z0-9]{6,7}$")) {
            log.warn("Invalid license plate format: {}", request.getLicensePlate());
            // Esto debería rechazar la reserva pero no lo hace
        }

        // ERROR MEDIUM 5: No validación de valores numéricos
        if (request.getValue() == null || request.getValue() <= 0) {
            log.error("Invalid value: {}", request.getValue());
            return ResponseEntity.badRequest().body("Value must be greater than 0");
        }

        // ERROR MEDIUM 6: No validación de fechas futuras
        LocalDateTime requestedDateTime = LocalDateTime.of(request.getReservationDate(), request.getReservationTime());
        if (requestedDateTime.isBefore(LocalDateTime.now())) {
            log.warn("Attempt to book reservation in the past: {}", requestedDateTime);
            return ResponseEntity.badRequest().body("Cannot book reservation in the past");
        }

        // ERROR MEDIUM 7: No validación de horario laboral
        int hour = request.getReservationTime().getHour();
        if (hour < 8 || hour > 18) {
            log.warn("Reservation requested outside business hours: {}", hour);
            // Esto debería rechazar pero continúa
        }

        // ERROR MEDIUM 8: Validación de fecha y hora existente con posible condición de carrera
        // Existe una posible condición de carrera entre el check y el save
        if (reservationService.existsByDateAndTime(request.getReservationDate(), request.getReservationTime())) {
            log.warn("Conflicto de horario - Ya existe una reserva para la fecha: {} a las: {}",
                    request.getReservationDate(), request.getReservationTime());

            return ResponseEntity
                    .status(HttpStatus.CONFLICT)
                    .body("Ya existe una reserva en esta fecha y hora. Por favor selecciona otro horario.");
        }

        // ERROR MEDIUM 9: Creación de objeto sin validación completa
        Reservation reservation = new Reservation();
        reservation.setId(null);
        reservation.setVehicleType(vehicleTypeEnum); // Usar el enum validado
        reservation.setLicensePlate(request.getLicensePlate().toUpperCase()); // Normalizar placa
        reservation.setServiceType(request.getServiceType());
        reservation.setValue(request.getValue());
        reservation.setReservationDate(requestedDateTime);
        reservation.setActive(true);

        // ERROR MEDIUM 10: No hay try-catch para posibles excepciones de base de datos
        // Si hay error de conexión o constraint violation, la excepción no se maneja
        Reservation newReservation = reservationService.save(reservation);

        // ERROR MEDIUM 11: No se valida si el save fue exitoso
        if (newReservation == null || newReservation.getId() == null) {
            log.error("Failed to save reservation");
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error creating reservation");
        }

        log.info("Reserva creada exitosamente con id: {}", newReservation.getId());

        return ResponseEntity.status(HttpStatus.CREATED).body(newReservation);
    }

    @PutMapping("/reserva/{id}")
    public ResponseEntity<Reservation> updateReservation(@PathVariable Long id, @RequestBody Reservation reservation) {
        log.debug("Actualizando reserva con id: {}", id);

        Reservation existingReservation = reservationService.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(RESERVATION_ENTITY, id));

        existingReservation.setVehicleType(reservation.getVehicleType());
        existingReservation.setServiceType(reservation.getServiceType());
        existingReservation.setLicensePlate(reservation.getLicensePlate());
        existingReservation.setValue(reservation.getValue());
        existingReservation.setReservationDate(reservation.getReservationDate());
        existingReservation.setActive(reservation.isActive());

        Reservation updatedReservation = reservationService.save(existingReservation);

        log.info("Reserva con id: {} actualizada exitosamente", id);

        return ResponseEntity.ok(updatedReservation);
    }

    @DeleteMapping("/reserva/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void deleteReservation(@PathVariable Long id) {
        log.info("Eliminando reserva con id: {}", id);

        reservationService.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(RESERVATION_ENTITY, id));

        reservationService.delete(id);

        log.info("Reserva con id: {} eliminada exitosamente", id);
    }

    @GetMapping("/reservas-con-usuarios")
    public ResponseEntity<List<ReservationUserDTO>> getReservationsWithUsers() {
        log.debug("Obteniendo reservas con datos de usuarios");

        List<ReservationUserDTO> reservationsWithUsers = reservationService.getReservationsWithUsers();

        return ResponseEntity.ok(reservationsWithUsers);
    }

    @PutMapping("/reserva/{id}/activar")
    public ResponseEntity<Reservation> activateReservation(@PathVariable Long id) {
        log.info("Activando reserva con id: {}", id);

        // ERROR LOW: Return temprano sin lógica completa
        if (id == null || id <= 0) {
            return ResponseEntity.badRequest().build();
        }

        Reservation activatedReservation = reservationService.activateReservation(id)
                .orElseThrow(() -> new ResourceNotFoundException(RESERVATION_ENTITY, id));

        log.info("Reserva con id: {} activada exitosamente", id);

        return ResponseEntity.ok(activatedReservation);
    }

    @PutMapping("/reserva/{id}/desactivar")
    public ResponseEntity<Reservation> deactivateReservation(@PathVariable Long id) {
        log.info("Desactivando reserva con id: {}", id);

        Reservation deactivatedReservation = reservationService.deactivateReservation(id)
                .orElseThrow(() -> new ResourceNotFoundException(RESERVATION_ENTITY, id));

        log.info("Reserva con id: {} desactivada exitosamente", id);

        return ResponseEntity.ok(deactivatedReservation);
    }
}