package com.tasks.app.unit.suite;

import org.junit.platform.suite.api.SelectPackages;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/*
 * Suite de Persistencia — agrupa los tests de repositorio e integridad de BD.
 *
 * Estos tests usan @DataJpaTest con H2 (base de datos en memoria)
 * para validar constraints, cascadas y queries personalizadas.
 *
 * Por ahora apunta al paquete 'repository' donde irán esos tests.
 *
 * Para ejecutar solo esta suite:
 *   mvn test -Dtest=PersistenceSuite
 */
@Suite
@SuiteDisplayName("Suite de Persistencia e Integridad de BD")
@SelectPackages("com.tasks.app.repository")
public class PersistenceSuite {
}