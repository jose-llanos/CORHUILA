package com.sgplab.backend.Iservice;

import com.sgplab.backend.model.entity.Equipo;
import java.util.List;

public interface IEquipoService {
    Equipo crearEquipo(Equipo equipo);
    Equipo obtenerEquipoPorId(Long id);
    List<Equipo> obtenerTodosLosEquipos();
    Equipo actualizarEquipo(Long id, Equipo equipoDetalles);

    void eliminarEquipo(Long id);
}

