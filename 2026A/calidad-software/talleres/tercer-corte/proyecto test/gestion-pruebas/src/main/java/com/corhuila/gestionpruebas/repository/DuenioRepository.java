package com.corhuila.gestionpruebas.repository;

import com.corhuila.gestionpruebas.model.Duenio;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface DuenioRepository extends JpaRepository<Duenio, Long> {
    // Aquí Spring Boot crea automáticamente los métodos save, findAll, delete, etc.
}