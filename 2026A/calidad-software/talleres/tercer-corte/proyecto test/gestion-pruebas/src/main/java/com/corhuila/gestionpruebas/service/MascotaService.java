package com.corhuila.gestionpruebas.service;

import com.corhuila.gestionpruebas.model.Mascota;
import com.corhuila.gestionpruebas.repository.CitaRepository;
import com.corhuila.gestionpruebas.repository.MascotaRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.util.List;

@Service
public class MascotaService {

    @Autowired
    private MascotaRepository mascotaRepository;

    @Autowired
    private CitaRepository citaRepository; // ✅ NUEVO

    public List<Mascota> obtenerTodas() {
        return mascotaRepository.findAll();
    }

    public Mascota guardar(Mascota mascota) {
        if (mascota.getNombre() == null || mascota.getNombre().isEmpty()) {
            throw new IllegalArgumentException("El nombre es requerido");
        }
        if (mascota.getEspecie() == null || mascota.getEspecie().isEmpty()) {
            throw new IllegalArgumentException("La especie es requerida");
        }
        return mascotaRepository.save(mascota);
    }

    public Mascota buscarPorId(Long id) {
        return mascotaRepository.findById(id).orElse(null);
    }

    @Transactional
    public void eliminar(Long id) {
        // ✅ Primero eliminar citas asociadas a esta mascota
        citaRepository.deleteAll(citaRepository.findByMascotaId(id));
        // ✅ Luego eliminar la mascota
        mascotaRepository.deleteById(id);
    }
}