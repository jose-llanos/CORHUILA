package com.corhuila.gestionpruebas.service;

import com.corhuila.gestionpruebas.model.Cita;
import com.corhuila.gestionpruebas.repository.CitaRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class CitaService {

    @Autowired
    private CitaRepository citaRepository;

    public List<Cita> obtenerTodas() {
        return citaRepository.findAll();
    }

    public Cita guardar(Cita cita) {
        if (cita.getMotivo() == null || cita.getMotivo().isEmpty()) {
            throw new IllegalArgumentException("El motivo es requerido");
        }
        if (cita.getFecha() == null) {
            throw new IllegalArgumentException("La fecha es requerida");
        }
        if (cita.getEstado() == null || cita.getEstado().isEmpty()) {
            cita.setEstado("PENDIENTE");
        }
        return citaRepository.save(cita);
    }

    public Cita buscarPorId(Long id) {
        return citaRepository.findById(id).orElse(null);
    }

    public void eliminar(Long id) {
        citaRepository.deleteById(id);
    }

    public Cita cambiarEstado(Long id, String nuevoEstado) {
        Cita cita = buscarPorId(id);
        if (cita == null) throw new IllegalArgumentException("Cita no encontrada");
        cita.setEstado(nuevoEstado);
        return citaRepository.save(cita);
    }
}