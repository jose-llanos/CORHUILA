package com.map.parking_project.controllers;

import com.map.parking_project.models.Tarifa;
import com.map.parking_project.services.ITarifaServices;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import com.map.parking_project.dto.TarifaDTO;

import java.util.List;
import java.util.Map; // 🔹 Importación añadida para manejar respuestas JSON

/**
 * Controlador REST encargado de gestionar las operaciones CRUD para las Tarifas.
 */
@RestController
@RequestMapping("/api/tarifas")
@CrossOrigin(origins = {"http://localhost:4200"})
public class TarifaRestController {

    @Autowired
    private ITarifaServices tarifaService;

    // 🔹 Constante para evitar duplicidad de Strings (SonarQube rule)
    private static final String ERROR_KEY = "error";

    /*
     *Obtiene la lista de todas las tarifas disponibles.
     *@return ResponseEntity con la lista de tarifas.
     */
    @GetMapping
    public ResponseEntity<List<Tarifa>> getAllTarifas() {
        return ResponseEntity.ok(tarifaService.findAll());
    }

    /*
     *Obtiene una tarifa específica por su ID.
     *@param id Identificador único de la tarifa.
     *@return ResponseEntity con la tarifa encontrada o código 404 si no existe.
     */
    @GetMapping("/{id}")
    public ResponseEntity<Tarifa> getTarifaById(@PathVariable Long id) {
        Tarifa tarifa = tarifaService.findById(id);
        return tarifa != null ? ResponseEntity.ok(tarifa) : ResponseEntity.notFound().build();
    }

    /**
     * Crea y guarda una nueva tarifa en el sistema.
     * @param tarifa Objeto con los datos de la nueva tarifa.
     * @return ResponseEntity con la tarifa creada (201) o un JSON con el error (400/500).
     */
    // 🔹 Crear nueva tarifa
    @PostMapping
    public ResponseEntity<Object> crearTarifa(@RequestBody TarifaDTO tarifaDTO) { // <-- ¡Recibe el DTO!
        try {
            // Validar los datos
            if (tarifaDTO.getTipoVehiculo() == null || tarifaDTO.getTarifaDiurna() == null || tarifaDTO.getTarifaNocturna() == null) {
                return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(Map.of(
                        ERROR_KEY, "Todos los campos son obligatorios."
                ));
            }

            // Mapeo: Convertir el DTO a la Entidad Tarifa
            Tarifa nuevaTarifa = new Tarifa();
            nuevaTarifa.setTipoVehiculo(tarifaDTO.getTipoVehiculo());
            nuevaTarifa.setTarifaDiurna(tarifaDTO.getTarifaDiurna());
            nuevaTarifa.setTarifaNocturna(tarifaDTO.getTarifaNocturna());
            nuevaTarifa.setImagen(tarifaDTO.getImagen()); // Si es nulo, no pasa nada

            // Guardar la tarifa en la base de datos
            Tarifa tarifaGuardada = tarifaService.save(nuevaTarifa);
            
            // Devolver 201 CREATED
            return ResponseEntity.status(HttpStatus.CREATED).body(tarifaGuardada);
            
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(Map.of(
                    ERROR_KEY, "Error al crear la tarifa: " + e.getMessage()
            ));
        }
    }

    /**
     * Actualiza los datos de una tarifa existente.
     * @param id Identificador de la tarifa a actualizar.
     * @param tarifaDetails Objeto con los nuevos datos de la tarifa.
     * @return ResponseEntity con la tarifa actualizada o código 404 si no se encuentra.
     */
    @PutMapping("/{id}")
    public ResponseEntity<Tarifa> updateTarifa(@PathVariable Long id, @RequestBody TarifaDTO tarifaDetails) { // <-- ¡Recibe el DTO!
        
        Tarifa tarifa = tarifaService.findById(id);
        
        if (tarifa == null) {
            return ResponseEntity.notFound().build();
        }

        // Mapeo seguro de datos del DTO a la Entidad encontrada
        tarifa.setTipoVehiculo(tarifaDetails.getTipoVehiculo());
        tarifa.setTarifaDiurna(tarifaDetails.getTarifaDiurna());
        tarifa.setTarifaNocturna(tarifaDetails.getTarifaNocturna());
        tarifa.setImagen(tarifaDetails.getImagen());

        return ResponseEntity.ok(tarifaService.save(tarifa));
    }

    /*
     *Elimina una tarifa del sistema mediante su ID.
     *@param id Identificador de la tarifa a eliminar.
     *@return ResponseEntity indicando éxito con código 204 (No Content).
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteTarifa(@PathVariable Long id) {
        tarifaService.delete(id);
        return ResponseEntity.noContent().build();
    }
}