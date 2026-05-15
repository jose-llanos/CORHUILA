package com.sgplab.backend.repository;

import com.sgplab.backend.model.entity.Prestamo;
import com.sgplab.backend.model.enums.EstadoPrestamo;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * Repositorio Spring Data JPA para {@link Prestamo}.
 *
 * @author SGP LAB Team
 */
@Repository
public interface IPrestamoRepository extends JpaRepository<Prestamo, Long> {

    /**
     * Indica si el usuario ya tiene un prestamo en el estado indicado.
     *
     * @param usuarioId id del usuario
     * @param estado    estado a comprobar
     * @return {@code true} si existe al menos uno, {@code false} en caso contrario.
     */
    boolean existsByUsuarioIdAndEstado(Long usuarioId, EstadoPrestamo estado);

    /**
     * Indica si el equipo aparece en algun prestamo en el estado indicado.
     *
     * @param equipoId id del equipo
     * @param estado   estado a comprobar
     * @return {@code true} si existe al menos uno, {@code false} en caso contrario.
     */
    boolean existsByEquipoIdAndEstado(Long equipoId, EstadoPrestamo estado);

    List<Prestamo> findByUsuarioId(Long usuarioId);
}
