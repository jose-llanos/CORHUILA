package com.sgplab.backend.repository;

import com.sgplab.backend.model.entity.Penalizacion;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface IPenalizacionRepository extends JpaRepository<Penalizacion, Long> {
    boolean existsByUsuarioId(Long usuarioId);
}