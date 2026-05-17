package com.autospark.migueljuliana.controllers;

import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.autospark.migueljuliana.services.ICarWashServiceService;
import com.autospark.migueljuliana.models.CarWashService;
import com.autospark.migueljuliana.exception.ResourceNotFoundException;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@CrossOrigin(origins = "http://localhost:4200", allowCredentials = "true")
@RestController
@RequestMapping("/autospark")
@RequiredArgsConstructor
public class ServiceController {

    private static final String SERVICE_ENTITY = "Service";

    private final ICarWashServiceService service;

    @GetMapping("/service")
    public ResponseEntity<List<CarWashService>> getAllServices() {
        log.debug("Obteniendo todos los servicios");
        List<CarWashService> services = service.findAll();
        return ResponseEntity.ok(services);
    }

    @GetMapping("/service/{id}")
    public ResponseEntity<CarWashService> getServiceById(@PathVariable Long id) {
        log.debug("Buscando servicio con id: {}", id);

        CarWashService carWashService = service.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(SERVICE_ENTITY, id));

        return ResponseEntity.ok(carWashService);
    }

    @PostMapping("/service")
    public ResponseEntity<CarWashService> createService(@RequestBody CarWashService carWashService) {
        log.info("Creando nuevo servicio - Nombre: {}, Precio: {}",
                carWashService.getName(),
                carWashService.getPrice());

        carWashService.setId(null);
        CarWashService newService = service.save(carWashService);

        log.info("Servicio creado exitosamente con id: {}", newService.getId());

        return ResponseEntity.status(HttpStatus.CREATED).body(newService);
    }

    @PutMapping("/service/{id}")
    public ResponseEntity<CarWashService> updateService(
            @PathVariable Long id,
            @RequestBody CarWashService carWashService) {

        log.debug("Actualizando servicio con id: {}", id);

        CarWashService existingService = service.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(SERVICE_ENTITY, id));

        existingService.setName(carWashService.getName());
        existingService.setDescription(carWashService.getDescription());
        existingService.setPrice(carWashService.getPrice());
        existingService.setActive(carWashService.isActive());
        existingService.setImageUrl(carWashService.getImageUrl());

        CarWashService updatedService = service.save(existingService);

        log.info("Servicio con id: {} actualizado exitosamente", id);

        return ResponseEntity.ok(updatedService);
    }

    @DeleteMapping("/service/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void deleteService(@PathVariable Long id) {
        log.info("Eliminando servicio con id: {}", id);
        service.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(SERVICE_ENTITY, id));

        service.delete(id);

        log.info("Servicio con id: {} eliminado exitosamente", id);
    }
}