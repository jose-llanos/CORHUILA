package com.autospark.migueljuliana.repositories;

import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;
import com.autospark.migueljuliana.models.Reservation;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

public interface IReservationRepository extends CrudRepository<Reservation, Long> {
    boolean existsByReservationDateBetween(LocalDateTime start, LocalDateTime end);

    @Query("SELECT DISTINCT CAST(r.reservationDate AS LocalDate) FROM Reservation r")
    List<LocalDate> findDistinctReservationDates();
}