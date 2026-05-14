package com.sgplab.backend.repository;

import com.sgplab.backend.model.entity.Equipo;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface IEquipoRepository extends JpaRepository<Equipo, Long> {
}