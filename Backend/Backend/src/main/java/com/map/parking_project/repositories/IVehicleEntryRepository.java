package com.map.parking_project.repositories;

import com.map.parking_project.models.VehicleEntry;
import org.springframework.data.jpa.repository.JpaRepository;

public interface IVehicleEntryRepository extends JpaRepository<VehicleEntry, Long> {
}
