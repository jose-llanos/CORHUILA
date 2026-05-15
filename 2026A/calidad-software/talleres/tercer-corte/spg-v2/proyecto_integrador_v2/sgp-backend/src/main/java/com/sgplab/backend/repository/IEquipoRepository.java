package com.sgplab.backend.repository;

import com.sgplab.backend.model.entity.Equipo;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

/**
 * Repositorio Spring Data JPA para {@link Equipo}.
 *
 * @author SGP LAB Team
 */
@Repository
public interface IEquipoRepository extends JpaRepository<Equipo, Long> {

    Optional<Equipo> findByCodigoInventario(String codigoInventario);

    boolean existsByCodigoInventario(String codigoInventario);
}
