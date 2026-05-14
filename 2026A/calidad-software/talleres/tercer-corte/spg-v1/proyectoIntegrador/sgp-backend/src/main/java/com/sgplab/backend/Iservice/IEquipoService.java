package com.sgplab.backend.Iservice;

import com.sgplab.backend.model.entity.Equipo;
import java.util.List;

public interface IEquipoService {
    // Métodos CRUD correctos
    Equipo crearEquipo(Equipo equipo);
    Equipo obtenerEquipoPorId(Long id);
    List<Equipo> obtenerTodosLosEquipos();
    Equipo actualizarEquipo(Long id, Equipo equipoDetalles);
    //void borrar_equipo_definitivamente_del_sistema_bd(Long id);

    void eliminarEquipo(Long id);
}

