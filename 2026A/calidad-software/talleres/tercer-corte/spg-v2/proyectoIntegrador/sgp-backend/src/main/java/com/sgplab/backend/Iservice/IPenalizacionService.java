package com.sgplab.backend.Iservice;

import com.sgplab.backend.model.entity.Penalizacion;
import java.util.List;

public interface IPenalizacionService {
    Penalizacion crearPenalizacion(Penalizacion penalizacion);
    Penalizacion obtenerPenalizacionPorId(Long id);
    List<Penalizacion> obtenerTodasLasPenalizaciones();
    Penalizacion actualizarPenalizacion(Long id, Penalizacion penalizacionDetalles);
    void eliminarPenalizacion(Long id);
    boolean isUsuarioPenalizado(Long usuarioId);
}