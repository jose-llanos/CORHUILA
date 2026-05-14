package com.sgplab.backend.service;

import com.sgplab.backend.Iservice.IEquipoService;
import com.sgplab.backend.model.entity.Equipo;
import com.sgplab.backend.repository.IEquipoRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class EquipoService implements IEquipoService {

    private final IEquipoRepository equipoRepository;

    public EquipoService(IEquipoRepository equipoRepository) {
        this.equipoRepository = equipoRepository;
    }

    @Override
    public Equipo crearEquipo(Equipo equipo) {
        return equipoRepository.save(equipo);
    }

    @Override
    public Equipo obtenerEquipoPorId(Long id) {
        return equipoRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Equipo no encontrado con ID: " + id));
    }

    @Override
    public List<Equipo> obtenerTodosLosEquipos() {
        return equipoRepository.findAll();
    }

    @Override
    public Equipo actualizarEquipo(Long id, Equipo equipoDetalles) {
        Equipo equipoExistente = obtenerEquipoPorId(id);

        equipoExistente.setNombre(equipoDetalles.getNombre());
        equipoExistente.setCodigoInventario(equipoDetalles.getCodigoInventario());
        equipoExistente.setCantidad(equipoDetalles.getCantidad());
        equipoExistente.setEstado(equipoDetalles.getEstado());

        return equipoRepository.save(equipoExistente);
    }
    @Override
    public void eliminarEquipo(Long id) {
        if (!equipoRepository.existsById(id)) {
            throw new RuntimeException("No se puede eliminar: El equipo con ID " + id + " no existe.");
        }
        try {
            equipoRepository.deleteById(id);
        } catch (Exception e) {
            // Manejo de errores de integridad (ej: el equipo está en un préstamo activo)
            throw new RuntimeException("Error al eliminar el equipo. Verifique que no tenga préstamos activos.");
        }
    }



}