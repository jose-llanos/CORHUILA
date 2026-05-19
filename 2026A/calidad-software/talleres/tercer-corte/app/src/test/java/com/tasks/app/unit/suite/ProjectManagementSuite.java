package com.tasks.app.unit.suite;

import com.tasks.app.unit.service.ProjectServiceTest;
import org.junit.platform.suite.api.SelectClasses;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/*
 * Suite de Gestión de Proyectos — agrupa los tests relacionados con:
 *   - Crear, listar, editar y eliminar proyectos
 *   - Invitar y remover miembros
 *   - Listar miembros del proyecto
 *
 * Para ejecutar solo esta suite:
 *   mvn test -Dtest=ProjectManagementSuite
 */
@Suite
@SuiteDisplayName("Suite de Gestión de Proyectos")
@SelectClasses({
        ProjectServiceTest.class
})
public class ProjectManagementSuite {
}