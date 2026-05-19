package com.tasks.app.repository;

import com.tasks.app.entity.Project;
import com.tasks.app.entity.Task;
import com.tasks.app.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface TaskRepository extends JpaRepository<Task, Long> {

    // RF-03.6: todas las tareas de un proyecto (el agrupamiento por estado se hace en el servicio)
    List<Task> findAllByProject(Project project);

    // RF-02.7: desasignar tareas cuando un miembro es removido del proyecto
    @Modifying
    @Query("UPDATE Task t SET t.assignedTo = null WHERE t.project = :project AND t.assignedTo = :user")
    void unassignTasksFromUserInProject(@Param("project") Project project, @Param("user") User user);
}
