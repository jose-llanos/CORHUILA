package com.sgplab.backend.repository;

import com.sgplab.backend.model.entity.Prestamo;
import com.sgplab.backend.model.enums.EstadoPrestamo;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface IPrestamoRepository extends JpaRepository<Prestamo, Long> {
    boolean existsByUsuarioIdAndEstado(Long usuarioId, EstadoPrestamo estado);
}