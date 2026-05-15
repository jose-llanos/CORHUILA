package com.autospark.migueljuliana.repositories;

import org.springframework.data.repository.CrudRepository;
import com.autospark.migueljuliana.models.Reservation;

import java.time.LocalDateTime;

public interface IReservationRepository extends CrudRepository<Reservation, Long> {
    boolean existsByReservationDateBetween(LocalDateTime start, LocalDateTime end);
}