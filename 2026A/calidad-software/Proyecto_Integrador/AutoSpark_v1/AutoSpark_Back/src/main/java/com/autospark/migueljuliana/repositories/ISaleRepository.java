package com.autospark.migueljuliana.repositories;

import com.autospark.migueljuliana.models.Sale;
import java.util.List;
import org.springframework.data.repository.CrudRepository;

public interface ISaleRepository extends CrudRepository<Sale, Long> {

    List<Sale> findByVehiclePlate(String vehiclePlate);
}