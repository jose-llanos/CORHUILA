package com.autospark.migueljuliana.services;

import java.time.LocalDate;
import java.time.LocalTime;
import java.util.List;
import java.util.Optional;

import com.autospark.migueljuliana.models.Reservation;
import com.autospark.migueljuliana.models.ReservationUserDTO;

public interface IReservationService {

    List<Reservation> findAll();

    Optional<Reservation> findById(Long id);

    Reservation save(Reservation reservation);

    void update(Reservation reservation, Long id);

    void delete(Long id);

    List<ReservationUserDTO> getReservationsWithUsers();

    Optional<Reservation> activateReservation(Long id);

    Optional<Reservation> deactivateReservation(Long id);

    boolean existsByDateAndTime(LocalDate date, LocalTime time);
}