package com.map.parking_project.services;

import com.map.parking_project.models.VehicleEntry;
import com.map.parking_project.repositories.IVehicleEntryRepository;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class VehicleEntryService implements IVehiculeEntryService{
    @Autowired
    private IVehicleEntryRepository repository;

    public VehicleEntry save(VehicleEntry entry) {
        return repository.save(entry);
    }
    public List<VehicleEntry> findAll() {
        return repository.findAll();
    }

    @Override
    @Transactional
    public void delete(Long id) {
        repository.deleteById(id); // Elimina un usuario por su ID
    }
}
