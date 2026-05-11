package com.corhuila.gestionpruebas.service;

import com.corhuila.gestionpruebas.model.Tratamiento;
import com.corhuila.gestionpruebas.repository.TratamientoRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class TratamientoService {

    @Autowired
    private TratamientoRepository tratamientoRepository;

    public List<Tratamiento> obtenerTodos() {
        return tratamientoRepository.findAll();
    }

    public Tratamiento guardar(Tratamiento tratamiento) {
        if (tratamiento.getDescripcion() == null || tratamiento.getDescripcion().isEmpty()) {
            throw new IllegalArgumentException("La descripción es requerida");
        }
        return tratamientoRepository.save(tratamiento);
    }

    public Tratamiento buscarPorId(Long id) {
        return tratamientoRepository.findById(id).orElse(null);
    }

    public void eliminar(Long id) {
        tratamientoRepository.deleteById(id);
    }
}