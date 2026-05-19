package com.map.parking_project.repositories;

import com.map.parking_project.models.Tarifa;
import org.springframework.data.repository.CrudRepository;

public interface ITarifaRepository extends CrudRepository<Tarifa, Long> {
    // Aquí puedes agregar métodos personalizados si es necesario
    // Por ejemplo, para buscar tarifas por nombre o tipo

}
