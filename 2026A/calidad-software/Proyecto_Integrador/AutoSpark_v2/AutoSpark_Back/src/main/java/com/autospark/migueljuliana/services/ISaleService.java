package com.autospark.migueljuliana.services;

import java.util.List;
import java.util.Optional;

import com.autospark.migueljuliana.models.Sale;

public interface ISaleService {

    List<Sale> findAll();

    Sale save(Sale sale);

    Optional<Sale> findById(Long id);

    void delete(Long id);

    Sale convertReservationToSale(Long reservationId);

    List<Sale> findByPlate(String plate);

    void deleteByPlate(String plate);
}