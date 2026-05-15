package com.sgplab.backend.controller;

import com.sgplab.backend.dto.request.PenalizacionRequest;
import com.sgplab.backend.dto.response.PenalizacionResponse;
import com.sgplab.backend.service.contract.IPenalizacionService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * Controlador REST para la gestion de penalizaciones.
 * Solo el administrador puede crear, editar y eliminar.
 *
 * @author SGP LAB Team
 */
@RestController
@RequestMapping("/api/penalizaciones")
public class PenalizacionController {

    private final IPenalizacionService penalizacionService;

    public PenalizacionController(IPenalizacionService penalizacionService) {
        this.penalizacionService = penalizacionService;
    }

    @GetMapping
    @PreAuthorize("hasRole('ADMINISTRADOR')")
    public ResponseEntity<List<PenalizacionResponse>> obtenerTodas() {
        return ResponseEntity.ok(penalizacionService.obtenerTodas());
    }

    @GetMapping("/{id}")
    public ResponseEntity<PenalizacionResponse> obtenerPorId(@PathVariable Long id) {
        return ResponseEntity.ok(penalizacionService.obtenerPorId(id));
    }

    @GetMapping("/usuario/{usuarioId}/activa")
    public ResponseEntity<Boolean> verificarPenalizacionActiva(@PathVariable Long usuarioId) {
        return ResponseEntity.ok(penalizacionService.tienePenalizacionActiva(usuarioId));
    }

    @PostMapping
    @PreAuthorize("hasRole('ADMINISTRADOR')")
    public ResponseEntity<PenalizacionResponse> crear(@Valid @RequestBody PenalizacionRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED).body(penalizacionService.crear(request));
    }

    @PutMapping("/{id}")
    @PreAuthorize("hasRole('ADMINISTRADOR')")
    public ResponseEntity<PenalizacionResponse> actualizar(@PathVariable Long id,
                                                           @Valid @RequestBody PenalizacionRequest request) {
        return ResponseEntity.ok(penalizacionService.actualizar(id, request));
    }

    @DeleteMapping("/{id}")
    @PreAuthorize("hasRole('ADMINISTRADOR')")
    public ResponseEntity<Void> eliminar(@PathVariable Long id) {
        penalizacionService.eliminar(id);
        return ResponseEntity.noContent().build();
    }
}
