package com.sgplab.backend.controller;

import com.sgplab.backend.model.entity.Penalizacion;
import com.sgplab.backend.Iservice.IPenalizacionService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/penalizaciones")
public class PenalizacionController {

    private final IPenalizacionService penalizacionService;

    public PenalizacionController(IPenalizacionService penalizacionService) {
        this.penalizacionService = penalizacionService;
    }

    @GetMapping
    public ResponseEntity<List<Penalizacion>> obtenerTodas() {
        return new ResponseEntity<>(penalizacionService.obtenerTodasLasPenalizaciones(), HttpStatus.OK);
    }

    @GetMapping("/{id}")
    public ResponseEntity<Penalizacion> obtenerPorId(@PathVariable Long id) {
        return new ResponseEntity<>(penalizacionService.obtenerPenalizacionPorId(id), HttpStatus.OK);
    }

    /* Endpoint adicional de negocio: Verifica si un usuario tiene una penalización activa */
    @GetMapping("/usuario/{usuarioId}/activa")
    public ResponseEntity<Boolean> verificarPenalizacionUsuario(@PathVariable Long usuarioId) {
        return new ResponseEntity<>(penalizacionService.isUsuarioPenalizado(usuarioId), HttpStatus.OK);
    }

    @PostMapping
    public ResponseEntity<Penalizacion> crearPenalizacion(@RequestBody Penalizacion penalizacion) {
        return new ResponseEntity<>(penalizacionService.crearPenalizacion(penalizacion), HttpStatus.CREATED);
    }

    @PutMapping("/{id}")
    public ResponseEntity<Penalizacion> actualizarPenalizacion(@PathVariable Long id, @RequestBody Penalizacion penalizacion) {
        return new ResponseEntity<>(penalizacionService.actualizarPenalizacion(id, penalizacion), HttpStatus.OK);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> eliminarPenalizacion(@PathVariable Long id) {
        penalizacionService.eliminarPenalizacion(id);
        return new ResponseEntity<>(HttpStatus.NO_CONTENT);
    }
}