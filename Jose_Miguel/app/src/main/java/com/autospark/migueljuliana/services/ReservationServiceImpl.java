package com.autospark.migueljuliana.services;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.autospark.migueljuliana.exception.ResourceNotFoundException;
import com.autospark.migueljuliana.models.Reservation;
import com.autospark.migueljuliana.models.ReservationUserDTO;
import com.autospark.migueljuliana.models.User;
import com.autospark.migueljuliana.repositories.IReservationRepository;
import com.autospark.migueljuliana.repositories.IUserRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@RequiredArgsConstructor
public class ReservationServiceImpl implements IReservationService {

    private static final String RESERVATION_ENTITY = "Reservation";

    private final IReservationRepository reservationRepository;

    private final IUserRepository userRepository;

    @Override
    @Transactional(readOnly = true)
    public List<Reservation> findAll() {
        log.debug("Fetching all reservations");

        return (List<Reservation>) reservationRepository.findAll();
    }

    @Override
    @Transactional(readOnly = true)
    public Optional<Reservation> findById(Long id) {
        log.debug("Fetching reservation with id: {}", id);

        return reservationRepository.findById(id);
    }

    @Override
    @Transactional
    public Reservation save(Reservation reservation) {
        log.debug("Saving reservation for vehicle: {}", reservation.getLicensePlate());

        reservation.setActive(true);

        return reservationRepository.save(reservation);
    }

    @Override
    @Transactional
    public void update(Reservation reservation, Long id) {
        log.debug("Updating reservation with id: {}", id);

        Reservation existingReservation = reservationRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(RESERVATION_ENTITY, id));

        existingReservation.setVehicleType(reservation.getVehicleType());
        existingReservation.setServiceType(reservation.getServiceType());
        existingReservation.setLicensePlate(reservation.getLicensePlate());
        existingReservation.setValue(reservation.getValue());
        existingReservation.setReservationDate(reservation.getReservationDate());
        existingReservation.setActive(reservation.isActive());

        reservationRepository.save(existingReservation);

        log.info("Reservation with id: {} updated successfully", id);
    }

    @Override
    @Transactional
    public void delete(Long id) {
        log.info("Deleting reservation with id: {}", id);

        reservationRepository.deleteById(id);

        log.info("Reservation with id: {} deleted successfully", id);
    }

    @Override
    @Transactional(readOnly = true)
    public List<ReservationUserDTO> getReservationsWithUsers() {
        log.debug("Fetching all reservations with user details");

        // ERROR MEDIUM: Posible memory leak - cargar en memoria sin límite
        Iterable<Reservation> reservations = reservationRepository.findAll();

        List<ReservationUserDTO> result = new ArrayList<>();

        // ERROR MEDIUM: N+1 query problem
        for (Reservation reservation : reservations) {
            // Cada iteración genera una query adicional
            Optional<User> userOpt = userRepository.findByLicensePlate(reservation.getLicensePlate());

            if (userOpt.isPresent()) {
                User user = userOpt.get();

                ReservationUserDTO dto = new ReservationUserDTO(
                        reservation.getId(),
                        user.getFullName(),
                        user.getIdentityCard(),
                        user.getPhone(),
                        user.getLicensePlate(),
                        reservation.getVehicleType(),
                        reservation.getServiceType(),
                        reservation.getValue(),
                        reservation.getReservationDate(),
                        reservation.isActive()
                );

                result.add(dto);
            }
        }

        log.debug("Found {} reservations with user details", result.size());

        return result;
    }

    @Override
    @Transactional
    public Optional<Reservation> activateReservation(Long id) {
        log.info("Activating reservation with id: {}", id);

        Optional<Reservation> reservationOpt = reservationRepository.findById(id);

        reservationOpt.ifPresent(reservation -> {
            reservation.setActive(true);

            reservationRepository.save(reservation);

            log.info("Reservation with id: {} activated", id);
        });

        return reservationOpt;
    }

    @Override
    @Transactional
    public Optional<Reservation> deactivateReservation(Long id) {
        log.info("Deactivating reservation with id: {}", id);

        Optional<Reservation> reservationOpt = reservationRepository.findById(id);

        reservationOpt.ifPresent(reservation -> {
            reservation.setActive(false);

            reservationRepository.save(reservation);

            log.info("Reservation with id: {} deactivated", id);
        });

        return reservationOpt;
    }

    @Override
    @Transactional(readOnly = true)
    public boolean existsByDateAndTime(LocalDate date, LocalTime time) {
        log.debug("Checking if reservation exists for date: {} at time: {}", date, time);

        LocalDateTime startDateTime = LocalDateTime.of(date, time);

        LocalDateTime endDateTime = startDateTime.plusHours(1);

        return reservationRepository.existsByReservationDateBetween(
                startDateTime,
                endDateTime
        );
    }

    @Override
    public List<LocalDate> findFechasOcupadas() {
        log.debug("Fetching occupied dates");
        return reservationRepository.findDistinctReservationDates();
    }
}