package com.tasks.app.repository;

import com.tasks.app.entity.Project;
import com.tasks.app.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface ProjectRepository extends JpaRepository<Project, Long> {

    // RF-02.2: proyectos donde el usuario es owner O tiene membresía activa
    @Query("""
            SELECT p FROM Project p
            WHERE p.owner = :user
               OR EXISTS (
                   SELECT pm FROM ProjectMember pm
                   WHERE pm.project = p AND pm.user = :user
               )
            """)
    List<Project> findAllAccessibleByUser(@Param("user") User user);

    // RF-02.1: validar nombre único por owner antes de crear
    boolean existsByNameAndOwner(String name, User owner);
}
