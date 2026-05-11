package com.corhuila.gestionpruebas.repository;

import com.corhuila.gestionpruebas.model.Mascota;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface MascotaRepository extends JpaRepository<Mascota, Long> {
    List<Mascota> findByDuenioId(Long duenioId);
}