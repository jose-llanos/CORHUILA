package com.sgplab.backend.service;

import com.sgplab.backend.Iservice.IPrestamoService;
import com.sgplab.backend.model.entity.Equipo;
import com.sgplab.backend.model.entity.Prestamo;
import com.sgplab.backend.model.enums.EstadoPrestamo;
import com.sgplab.backend.repository.IEquipoRepository;
import com.sgplab.backend.repository.IPrestamoRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public class PrestamoService implements IPrestamoService {

    private final IPrestamoRepository prestamoRepository;
    private final IEquipoRepository equipoRepository;

    public PrestamoService(IPrestamoRepository prestamoRepository, IEquipoRepository equipoRepository) {
        this.prestamoRepository = prestamoRepository;
        this.equipoRepository = equipoRepository;
    }

    @Override
    @Transactional
    public Prestamo crearPrestamo(Prestamo prestamo) {
        // 1. Validaciones básicas de entrada
        if (prestamo.getUsuario() == null || prestamo.getUsuario().getId() == null) {
            throw new IllegalArgumentException("El préstamo debe tener un usuario asociado válido.");
        }
        if (prestamo.getEquipo() == null || prestamo.getEquipo().getId() == null) {
            throw new IllegalArgumentException("El préstamo debe tener un equipo asociado válido.");
        }

        // 2. LÓGICA DE NEGOCIO: Validar que el usuario no tenga ya un préstamo ACTIVO
        boolean tienePrestamoActivo = prestamoRepository.existsByUsuarioIdAndEstado(
                prestamo.getUsuario().getId(),
                EstadoPrestamo.ACTIVO
        );

        if (tienePrestamoActivo) {
            throw new IllegalStateException("El usuario ya tiene un préstamo activo y debe devolverlo antes de solicitar otro.");
        }

        // 3. Validar existencia del equipo y stock
        Equipo equipo = equipoRepository.findById(prestamo.getEquipo().getId())
                .orElseThrow(() -> new RuntimeException("El equipo solicitado no existe."));

        if (equipo.getCantidad() <= 0) {
            throw new IllegalStateException("No hay stock disponible para el equipo: " + equipo.getNombre());
        }

        // 4. Descontar del inventario y actualizar el equipo
        equipo.setCantidad(equipo.getCantidad() - 1);
        equipoRepository.save(equipo);

        // 5. Configurar el préstamo y guardar
        prestamo.setEquipo(equipo);
        prestamo.setEstado(EstadoPrestamo.ACTIVO); // Aseguramos que inicie como ACTIVO

        return prestamoRepository.save(prestamo);
    }

    @Override
    public Prestamo obtenerPrestamoPorId(Long id) {
        return prestamoRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Préstamo no encontrado con ID: " + id));
    }

    @Override
    public List<Prestamo> obtenerTodosLosPrestamos() {
        return prestamoRepository.findAll();
    }

    @Override
    @Transactional
    public Prestamo actualizarPrestamo(Long id, Prestamo prestamoDetalles) {
        Prestamo prestamoExistente = obtenerPrestamoPorId(id);

        // Si el estado cambia de ACTIVO a DEVUELTO, devolvemos el stock al inventario
        if (prestamoExistente.getEstado() == EstadoPrestamo.ACTIVO &&
                prestamoDetalles.getEstado() == EstadoPrestamo.DEVUELTO) {

            Equipo equipo = prestamoExistente.getEquipo();
            equipo.setCantidad(equipo.getCantidad() + 1);
            equipoRepository.save(equipo);
        }

        prestamoExistente.setFechaInicio(prestamoDetalles.getFechaInicio());
        prestamoExistente.setFechaFin(prestamoDetalles.getFechaFin());
        prestamoExistente.setEstado(prestamoDetalles.getEstado());

        return prestamoRepository.save(prestamoExistente);
    }

    @Override
    @Transactional
    public void eliminarPrestamo(Long id) {
        if (!prestamoRepository.existsById(id)) {
            throw new RuntimeException("No se puede eliminar: El préstamo con ID " + id + " no existe.");
        }
        prestamoRepository.deleteById(id);
    }
}