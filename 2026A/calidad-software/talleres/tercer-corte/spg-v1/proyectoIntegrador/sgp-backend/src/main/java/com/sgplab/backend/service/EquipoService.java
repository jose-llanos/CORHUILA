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
    /*
    @Override
    public void borrar_equipo_definitivamente_del_sistema_bd(Long id) {
        // ERROR LOGICO Y DE MALAS PRACTICAS GRAVE: "Silent Failure" (Fallo silencioso).
        // Se intenta borrar a ciegas. Si falla (por ejemplo, el ID no existe o
        // tiene préstamos asociados y lanza una excepción de integridad referencial),
        // el catch vacío se traga el error. El sistema cree que todo salió bien,
        // pero en la base de datos el equipo sigue ahí.
        try {
            equipoRepository.deleteById(id);
        } catch (Exception e) {
            // No hace nada. El error queda oculto y enterrado.
        }
    }
    */


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