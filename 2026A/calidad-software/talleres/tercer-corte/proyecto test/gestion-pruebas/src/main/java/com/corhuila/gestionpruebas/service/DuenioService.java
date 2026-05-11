package com.corhuila.gestionpruebas.service;

import com.corhuila.gestionpruebas.model.Duenio;
import com.corhuila.gestionpruebas.model.Mascota;
import com.corhuila.gestionpruebas.repository.CitaRepository;
import com.corhuila.gestionpruebas.repository.DuenioRepository;
import com.corhuila.gestionpruebas.repository.MascotaRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.util.List;

@Service
public class DuenioService {

    @Autowired
    private DuenioRepository duenioRepository;

    @Autowired
    private MascotaRepository mascotaRepository; // ✅ NUEVO

    @Autowired
    private CitaRepository citaRepository; // ✅ NUEVO

    public List<Duenio> obtenerTodos() {
        return duenioRepository.findAll();
    }

    public Duenio guardar(Duenio duenio) {
        if (duenio.getNombre() == null || duenio.getNombre().isEmpty()) {
            throw new IllegalArgumentException("El nombre es requerido");
        }
        return duenioRepository.save(duenio);
    }

    public Duenio buscarPorId(Long id) {
        return duenioRepository.findById(id).orElse(null);
    }

    @Transactional
    public void eliminar(Long id) {
        // ✅ 1. Obtener todas las mascotas del dueño
        List<Mascota> mascotas = mascotaRepository.findByDuenioId(id);
        // ✅ 2. Por cada mascota, eliminar sus citas primero
        for (Mascota mascota : mascotas) {
            citaRepository.deleteAll(citaRepository.findByMascotaId(mascota.getId()));
        }
        // ✅ 3. Eliminar todas las mascotas del dueño
        mascotaRepository.deleteAll(mascotas);
        // ✅ 4. Finalmente eliminar el dueño
        duenioRepository.deleteById(id);
    }
}