package com.map.parking.selenium.tests;

import com.map.parking.selenium.config.SeleniumConfig;
import com.map.parking.selenium.pages.AdminPage;
import com.map.parking.selenium.pages.HomePage;
import com.map.parking.selenium.pages.LoginPage;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Cinco escenarios funcionales críticos (ISO/IEC 29119 – pruebas de sistema UI).
 */
@DisplayName("MAP Parking - Pruebas funcionales UI")
class ParkingFunctionalTest extends BaseUiTest {

    @Test
    @DisplayName("01 - La página de inicio muestra el mensaje de bienvenida")
    void homePageShowsWelcomeMessage() {
        HomePage home = new HomePage(driver).open();

        assertTrue(home.getWelcomeText().contains("MAP PARKING"),
                "El título de bienvenida debe mencionar MAP PARKING");
    }

    @Test
    @DisplayName("02 - Desde home se navega al formulario de login")
    void navigateFromHomeToLogin() {
        LoginPage login = new HomePage(driver).open().goToLogin();

        assertTrue(driver.getCurrentUrl().contains("/login"),
                "La URL debe incluir /login");
        login.enterEmail("probe@test.com");
        assertTrue(driver.findElement(org.openqa.selenium.By.name("email")).isDisplayed());
    }

    @Test
    @DisplayName("03 - Login con credenciales inválidas muestra error")
    void loginWithInvalidCredentialsShowsError() {
        LoginPage login = new LoginPage(driver).open()
                .login("noexiste@test.com", "wrong-password");

        assertEquals("Usuario o contraseña incorrectos", login.waitForErrorMessage().trim());
    }

    @Test
    @DisplayName("04 - Login administrador redirige al panel admin")
    void loginAsAdminRedirectsToAdminPanel() {
        new LoginPage(driver).open()
                .login(SeleniumConfig.TEST_ADMIN_EMAIL, SeleniumConfig.TEST_ADMIN_PASSWORD);

        assertTrue(new AdminPage(driver).isAdminPanelVisible(),
                "Debe mostrarse el panel de administrador");
    }

    @Test
    @DisplayName("05 - Desde home se accede a la página de servicios")
    void navigateFromHomeToServicios() {
        var servicios = new HomePage(driver).open().goToServicios();

        assertTrue(driver.getCurrentUrl().contains("/servicios"));
        assertEquals("Nuestros Servicios", servicios.getPageTitle());
    }

    @Test
    @DisplayName("06 - Desde login se accede a recuperar contraseña")
    void navigateFromLoginToRecoverPassword() {
        var recover = new LoginPage(driver).open().goToRecoverPassword();

        assertTrue(driver.getCurrentUrl().contains("/recuperarcontrasenia"));
        assertEquals("Recuperar Contraseña", recover.getPageTitle());
        assertTrue(recover.isEmailFieldVisible());
    }

    @Test
    @DisplayName("07 - Desde login se accede al registro")
    void navigateFromLoginToRegister() {
        var register = new LoginPage(driver).open().goToRegister();

        assertTrue(driver.getCurrentUrl().contains("/login/register"));
        assertEquals("REGISTRO", register.getPageTitle());
    }
}
