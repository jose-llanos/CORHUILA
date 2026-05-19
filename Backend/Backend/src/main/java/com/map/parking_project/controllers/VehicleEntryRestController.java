package com.map.parking_project.controllers;

import com.map.parking_project.models.VehicleEntry;
import com.map.parking_project.services.VehicleEntryService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import com.map.parking_project.dto.VehicleEntryDTO;
import org.springframework.http.ResponseEntity;
import java.util.List;

@RestController
@RequestMapping("/api/ingresos")
@CrossOrigin(origins = "http://localhost:4200")
public class VehicleEntryRestController {

    @Autowired
    private VehicleEntryService service;

    @PostMapping
    public ResponseEntity<VehicleEntry> registrarIngreso(@RequestBody VehicleEntryDTO entryDTO) {
        
        // 1. Instanciar la entidad real
        VehicleEntry entryEntidad = new VehicleEntry();
        
        // 2. Pasar los 4 datos del DTO a la Entidad
        entryEntidad.setPlaca(entryDTO.getPlaca());
        entryEntidad.setTipoVehiculo(entryDTO.getTipoVehiculo());
        entryEntidad.setUbicacion(entryDTO.getUbicacion());
        entryEntidad.setHoraIngreso(entryDTO.getHoraIngreso());
        
        // Nota: No necesitamos hacer 'setFechaIngreso' porque en tu modelo 
        // ya lo tienes inicializado por defecto con LocalDate.now().

        // 3. Guardar en la base de datos a través del servicio
        VehicleEntry guardado = service.save(entryEntidad);
        
        // 4. Retornar la entidad guardada con el código 201 (Created)
        return ResponseEntity.status(HttpStatus.CREATED).body(guardado);
    }

    @GetMapping
    public List<VehicleEntry> listarIngresos() {
        return service.findAll();
    }

    @DeleteMapping("/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void delete(@PathVariable Long id) {
        service.delete(id);
    }
}