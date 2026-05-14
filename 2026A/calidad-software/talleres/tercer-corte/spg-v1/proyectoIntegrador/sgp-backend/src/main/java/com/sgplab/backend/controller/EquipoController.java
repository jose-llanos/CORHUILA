package com.sgplab.backend.controller;

import com.sgplab.backend.model.entity.Equipo;
import com.sgplab.backend.Iservice.IEquipoService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/equipos")
public class EquipoController {

    private final IEquipoService equipoService;

    public EquipoController(IEquipoService equipoService) {
        this.equipoService = equipoService;
    }

    @GetMapping
    public ResponseEntity<List<Equipo>> obtenerTodos() {
        return new ResponseEntity<>(equipoService.obtenerTodosLosEquipos(), HttpStatus.OK);
    }

    @GetMapping("/{id}")
    public ResponseEntity<Equipo> obtenerPorId(@PathVariable Long id) {
        return new ResponseEntity<>(equipoService.obtenerEquipoPorId(id), HttpStatus.OK);
    }

    @PostMapping
    public ResponseEntity<Equipo> crearEquipo(@RequestBody Equipo equipo) {
        return new ResponseEntity<>(equipoService.crearEquipo(equipo), HttpStatus.CREATED);
    }

    @PutMapping("/{id}")
    public ResponseEntity<Equipo> actualizarEquipo(@PathVariable Long id, @RequestBody Equipo equipo) {
        return new ResponseEntity<>(equipoService.actualizarEquipo(id, equipo), HttpStatus.OK);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> eliminarEquipo(@PathVariable Long id) {
        equipoService.eliminarEquipo(id);
        return new ResponseEntity<>(HttpStatus.NO_CONTENT);
    }

    /* @DeleteMapping("/{id}")
    public ResponseEntity<Void> eliminarEquipo(@PathVariable Long id) {
        // Se solucionó con la estrategia de: "Refactorización de Nomenclatura (Clean Code)".
        // Se reemplazó el nombre snake_case y excesivamente largo por el estándar camelCase 'eliminarEquipo',
        // haciendo coincidir el llamado con la interfaz corregida IEquipoService.
        equipoService.eliminarEquipo(id);
        return new ResponseEntity<>(HttpStatus.NO_CONTENT);
    }
    */
}