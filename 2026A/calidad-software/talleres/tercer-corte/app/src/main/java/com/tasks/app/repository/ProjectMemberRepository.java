package com.tasks.app.repository;

import com.tasks.app.entity.Project;
import com.tasks.app.entity.ProjectMember;
import com.tasks.app.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface ProjectMemberRepository extends JpaRepository<ProjectMember, Long> {

    // RF-02.6 / RF-02.7 / RF-03.5: verificar si un usuario es miembro del proyecto
    boolean existsByProjectAndUser(Project project, User user);

    // RF-02.7: obtener membresía para eliminarla
    Optional<ProjectMember> findByProjectAndUser(Project project, User user);

    // RF-02.8: listar todos los miembros de un proyecto
    List<ProjectMember> findAllByProject(Project project);
}
