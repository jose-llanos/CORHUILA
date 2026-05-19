package edu.calidadsoftware.taskmanager.task;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

/**
 * Repositorio JPA para operaciones CRUD de Task.
 *
 * Spring Data JPA genera la implementación automáticamente.
 */
public interface TaskRepository extends JpaRepository<Task, Long> {

    List<Task> findByStatus(TaskStatus status);
}
