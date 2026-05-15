package com.autospark.migueljuliana.repositories;

import org.springframework.data.repository.CrudRepository;
import com.autospark.migueljuliana.models.CarWashService;

public interface ICarWashServiceRepository extends CrudRepository<CarWashService, Long> {
}