package com.tasks.app.unit.suite;

import com.tasks.app.unit.security.JwtServiceTest;
import com.tasks.app.unit.service.UserServiceTest;
import org.junit.platform.suite.api.SelectClasses;
import org.junit.platform.suite.api.Suite;
import org.junit.platform.suite.api.SuiteDisplayName;

/*
 * Suite de Autenticación — agrupa los tests relacionados con:
 *   - Registro e inicio de sesión de usuarios (UserService)
 *   - Generación y validación de tokens JWT (JwtService)
 *
 * Para ejecutar solo esta suite:
 *   mvn test -Dtest=AuthenticationSuite
 */
@Suite
@SuiteDisplayName("Suite de Autenticación y JWT")
@SelectClasses({
        UserServiceTest.class,
        JwtServiceTest.class
})
public class AuthenticationSuite {
}