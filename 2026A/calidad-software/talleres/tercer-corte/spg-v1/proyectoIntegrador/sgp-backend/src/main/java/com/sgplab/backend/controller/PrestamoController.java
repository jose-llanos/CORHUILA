package com.sgplab.backend.controller;

import com.sgplab.backend.model.entity.Prestamo;
import com.sgplab.backend.Iservice.IPrestamoService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/prestamos")
public class PrestamoController {

    private final IPrestamoService prestamoService;

    public PrestamoController(IPrestamoService prestamoService) {
        this.prestamoService = prestamoService;
    }

    @GetMapping
    public ResponseEntity<List<Prestamo>> obtenerTodos() {
        return new ResponseEntity<>(prestamoService.obtenerTodosLosPrestamos(), HttpStatus.OK);
    }

    @GetMapping("/{id}")
    public ResponseEntity<Prestamo> obtenerPorId(@PathVariable Long id) {
        return new ResponseEntity<>(prestamoService.obtenerPrestamoPorId(id), HttpStatus.OK);
    }

    @PostMapping
    public ResponseEntity<Prestamo> crearPrestamo(@RequestBody Prestamo prestamo) {
        Prestamo nuevoPrestamo = prestamoService.crearPrestamo(prestamo);
        return new ResponseEntity<>(nuevoPrestamo, HttpStatus.CREATED);
    }


    @PutMapping("/{id}")
    public ResponseEntity<Prestamo> actualizar(@PathVariable Long id, @RequestBody Prestamo prestamo) {
        return new ResponseEntity<>(prestamoService.actualizarPrestamo(id, prestamo), HttpStatus.OK);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> eliminar(@PathVariable Long id) {
        prestamoService.eliminarPrestamo(id);
        return new ResponseEntity<>(HttpStatus.NO_CONTENT);
    }
}