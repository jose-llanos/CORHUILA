package com.map.parking_project.repositories;

import com.map.parking_project.models.MapServices;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface IMapServiceRepository extends CrudRepository <MapServices, Long> {


}
