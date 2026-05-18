package com.autospark.migueljuliana.controllers;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.autospark.migueljuliana.services.IReservationService;
import com.autospark.migueljuliana.models.Reservation;
import com.autospark.migueljuliana.models.ReservationRequestDTO;
import com.autospark.migueljuliana.models.ReservationUserDTO;
import com.autospark.migueljuliana.models.VehicleType;
import com.autospark.migueljuliana.exception.ResourceNotFoundException;

import java.time.LocalDate;
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


    @GetMapping("/reserva/fechas-ocupadas")
    public ResponseEntity<List<LocalDate>> getFechasOcupadas() {
        log.debug("Obteniendo fechas ocupadas");
        List<LocalDate> fechasOcupadas = reservationService.findFechasOcupadas();
        return ResponseEntity.ok(fechasOcupadas);
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

        // VALIDACIÓN: Verificar que la hora no sea null
        if (request.getReservationTime() == null) {
            String errorMessage = "La hora de reserva es obligatoria. Por favor selecciona un horario.";
            log.warn("Intento de crear reserva sin especificar hora: {}", request);
            return ResponseEntity
                    .status(HttpStatus.BAD_REQUEST)
                    .body(errorMessage);
        }

        // ✅ ADD THIS CHECK - Verificar si el horario ya está ocupado
        if (reservationService.existsByDateAndTime(request.getReservationDate(), request.getReservationTime())) {
            String errorMessage = "Ya existe una reserva en esta fecha y hora. Por favor selecciona otro horario.";
            log.warn("Conflicto de horario - Intento de reserva duplicada para fecha: {} a las: {}",
                    request.getReservationDate(), request.getReservationTime());
            return ResponseEntity
                    .status(HttpStatus.CONFLICT)
                    .body(errorMessage);
        }

        log.info("Creando nueva reserva - Vehículo: {}, Servicio: {}, Placa: {}, Valor: {}, Fecha: {}, Hora: {}",
                request.getVehicleType(),
                request.getServiceType(),
                request.getLicensePlate(),
                request.getValue(),
                request.getReservationDate(),
                request.getReservationTime());

        // Crear objeto Reservation
        Reservation reservation = new Reservation();
        reservation.setId(null);
        reservation.setVehicleType(VehicleType.valueOf(request.getVehicleType()));
        reservation.setLicensePlate(request.getLicensePlate());
        reservation.setServiceType(request.getServiceType());
        reservation.setValue(request.getValue());
        reservation.setReservationDate(LocalDateTime.of(request.getReservationDate(), request.getReservationTime()));
        reservation.setActive(true);

        try {
            Reservation newReservation = reservationService.save(reservation);
            log.info("Reserva creada exitosamente con id: {}", newReservation.getId());
            return ResponseEntity.status(HttpStatus.CREATED).body(newReservation);

        } catch (Exception e) {
            log.error("Error inesperado al crear reserva: {}", e.getMessage(), e);
            return ResponseEntity
                    .status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error al crear la reserva: " + e.getMessage());
        }
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

        try {
            Reservation updatedReservation = reservationService.save(existingReservation);
            log.info("Reserva con id: {} actualizada exitosamente", id);
            return ResponseEntity.ok(updatedReservation);

        } catch (DataIntegrityViolationException e) {
            log.warn("Conflicto de horario al actualizar reserva {}: {}", id, e.getMessage());
            return ResponseEntity
                    .status(HttpStatus.CONFLICT)
                    .body(null);
        }
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