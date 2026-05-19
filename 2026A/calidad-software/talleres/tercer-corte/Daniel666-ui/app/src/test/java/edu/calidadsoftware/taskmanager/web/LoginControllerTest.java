package edu.calidadsoftware.taskmanager.web;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.security.core.Authentication;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.when;

/**
 * Pruebas unitarias para LoginController.
 */
@DisplayName("LoginController")
class LoginControllerTest {

    @Test
    @DisplayName("Si no hay autenticación, muestra la vista login")
    void login_view() {
        LoginController controller = new LoginController();
        assertEquals("login", controller.login(null));
    }

    @Test
    @DisplayName("Si hay autenticación, redirige al dashboard")
    void login_redirectsWhenAuthenticated() {
        LoginController controller = new LoginController();
        Authentication auth = Mockito.mock(Authentication.class);
        when(auth.isAuthenticated()).thenReturn(true);
        assertEquals("redirect:/dashboard", controller.login(auth));
    }
}

