package com.sgplab.backend.repository;

import com.sgplab.backend.model.entity.Penalizacion;
import com.sgplab.backend.model.enums.EstadoPenalizacion;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * Repositorio Spring Data JPA para {@link Penalizacion}.
 *
 * @author SGP LAB Team
 */
@Repository
public interface IPenalizacionRepository extends JpaRepository<Penalizacion, Long> {

    boolean existsByUsuarioIdAndEstado(Long usuarioId, EstadoPenalizacion estado);

    List<Penalizacion> findByUsuarioId(Long usuarioId);
}
