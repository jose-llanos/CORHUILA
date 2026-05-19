package com.tasks.app.unit.suite;

import com.tasks.app.unit.service.TaskServiceTest;
import org.junit.platform.suite.api.SelectClasses;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/*
 * Suite de Gestión de Tareas — agrupa los tests relacionados con:
 *   - Crear, editar y eliminar tareas
 *   - Cambiar el estado de una tarea
 *   - Asignar y desasignar tareas a miembros
 *   - Listar tareas de un proyecto
 *
 * Para ejecutar solo esta suite:
 *   mvn test -Dtest=TaskManagementSuite
 */
@Suite
@SuiteDisplayName("Suite de Gestión de Tareas")
@SelectClasses({
        TaskServiceTest.class
})
public class TaskManagementSuite {
}