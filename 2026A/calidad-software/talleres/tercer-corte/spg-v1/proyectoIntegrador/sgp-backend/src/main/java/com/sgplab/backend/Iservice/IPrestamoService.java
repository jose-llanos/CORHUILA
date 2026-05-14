package com.sgplab.backend.Iservice;

import com.sgplab.backend.model.entity.Prestamo;
import java.util.List;

public interface IPrestamoService {
    Prestamo crearPrestamo(Prestamo prestamo);
    Prestamo obtenerPrestamoPorId(Long id);
    List<Prestamo> obtenerTodosLosPrestamos();
    Prestamo actualizarPrestamo(Long id, Prestamo prestamoDetalles);
    void eliminarPrestamo(Long id);
}