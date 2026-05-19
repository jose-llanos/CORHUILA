package edu.calidadsoftware.taskmanager.user;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

/**
 * Repositorio JPA para usuarios.
 *
 * Se utiliza para autenticación (búsqueda por username) y validaciones de duplicados.
 */
public interface UserRepository extends JpaRepository<User, Long> {

    Optional<User> findByUsername(String username);

    boolean existsByUsername(String username);

    boolean existsByEmail(String email);
}
