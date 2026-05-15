package com.sgplab.backend.repository;

import com.sgplab.backend.model.entity.Usuario;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

/**
 * Repositorio Spring Data JPA para {@link Usuario}.
 *
 * @author SGP LAB Team
 */
@Repository
public interface IUsuarioRepository extends JpaRepository<Usuario, Long> {

    /**
     * Localiza un usuario por su email.
     *
     * @param email correo electronico (case-sensitive)
     * @return Optional con el usuario si existe, vacio en caso contrario.
     */
    Optional<Usuario> findByEmail(String email);

    /**
     * Verifica si ya existe un usuario con ese email.
     *
     * @param email correo electronico
     * @return {@code true} si existe, {@code false} en caso contrario.
     */
    boolean existsByEmail(String email);
}
