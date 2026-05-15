package com.autospark.migueljuliana.services;

import java.util.List;
import java.util.Optional;

import com.autospark.migueljuliana.exception.ResourceNotFoundException;
import com.autospark.migueljuliana.models.CarWashService;
import com.autospark.migueljuliana.repositories.ICarWashServiceRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Slf4j
@Service
@RequiredArgsConstructor
public class CarWashServiceServiceImpl implements ICarWashServiceService {

    private static final String SERVICE_ENTITY = "Service";

    private final ICarWashServiceRepository repository;

    @Override
    @Transactional(readOnly = true)
    public List<CarWashService> findAll() {
        log.debug("Fetching all car wash services");
        return (List<CarWashService>) repository.findAll();
    }

    @Override
    @Transactional(readOnly = true)
    public Optional<CarWashService> findById(Long id) {
        log.debug("Fetching car wash service with id: {}", id);
        return repository.findById(id);
    }

    @Override
    @Transactional
    public CarWashService save(CarWashService carWashService) {
        log.debug("Saving car wash service: {}", carWashService.getName());
        return repository.save(carWashService);
    }

    @Override
    @Transactional
    public void update(CarWashService carWashService, Long id) {
        log.debug("Updating car wash service with id: {}", id);

        CarWashService existingService = repository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(SERVICE_ENTITY, id));

        existingService.setName(carWashService.getName());
        existingService.setDescription(carWashService.getDescription());
        existingService.setPrice(carWashService.getPrice());
        existingService.setActive(carWashService.isActive());
        existingService.setImageUrl(carWashService.getImageUrl());

        repository.save(existingService);

        log.info("Car wash service with id: {} updated successfully", id);
    }

    @Override
    @Transactional
    public void delete(Long id) {
        log.info("Deleting car wash service with id: {}", id);

        repository.deleteById(id);

        log.info("Car wash service with id: {} deleted successfully", id);
    }
}