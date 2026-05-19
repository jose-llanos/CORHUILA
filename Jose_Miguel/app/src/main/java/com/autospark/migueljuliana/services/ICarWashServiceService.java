package com.autospark.migueljuliana.services;

import java.util.List;
import java.util.Optional;

import com.autospark.migueljuliana.models.CarWashService;

public interface ICarWashServiceService {

    List<CarWashService> findAll();

    Optional<CarWashService> findById(Long id);

    CarWashService save(CarWashService carWashService);

    void update(CarWashService carWashService, Long id);

    void delete(Long id);
}