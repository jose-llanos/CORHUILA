package com.sgplab.backend.service;

import com.sgplab.backend.Iservice.IPenalizacionService;
import com.sgplab.backend.model.entity.Penalizacion;
import com.sgplab.backend.repository.IPenalizacionRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class PenalizacionService implements IPenalizacionService {

    private final IPenalizacionRepository penalizacionRepository;

    public PenalizacionService(IPenalizacionRepository penalizacionRepository) {
        this.penalizacionRepository = penalizacionRepository;
    }

    @Override
    public Penalizacion crearPenalizacion(Penalizacion penalizacion) {
        if (penalizacion.getUsuario() == null || penalizacion.getUsuario().getId() == null) {
            throw new IllegalArgumentException("La penalización debe estar asociada a un usuario válido.");
        }
        return penalizacionRepository.save(penalizacion);
    }

    @Override
    public Penalizacion obtenerPenalizacionPorId(Long id) {
        return penalizacionRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Penalización no encontrada con ID: " + id));
    }

    @Override
    public List<Penalizacion> obtenerTodasLasPenalizaciones() {
        return penalizacionRepository.findAll();
    }

    @Override
    public Penalizacion actualizarPenalizacion(Long id, Penalizacion penalizacionDetalles) {
        Penalizacion penalizacionExistente = obtenerPenalizacionPorId(id);

        penalizacionExistente.setMotivo(penalizacionDetalles.getMotivo());
        penalizacionExistente.setFechaInicio(penalizacionDetalles.getFechaInicio());
        penalizacionExistente.setFechaFin(penalizacionDetalles.getFechaFin());
        penalizacionExistente.setEstado(penalizacionDetalles.getEstado());

        return penalizacionRepository.save(penalizacionExistente);
    }

    @Override
    public void eliminarPenalizacion(Long id) {
        if (!penalizacionRepository.existsById(id)) {
            throw new RuntimeException("No se puede eliminar: La penalización con ID " + id + " no existe.");
        }
        penalizacionRepository.deleteById(id);
    }

    @Override
    public boolean isUsuarioPenalizado(Long usuarioId) {
        return penalizacionRepository.existsByUsuarioId(usuarioId);
    }
}