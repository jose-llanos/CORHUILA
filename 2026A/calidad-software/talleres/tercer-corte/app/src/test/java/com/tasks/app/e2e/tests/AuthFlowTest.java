package com.tasks.app.e2e.tests;
 
import com.tasks.app.e2e.config.BaseE2ETest;
import com.tasks.app.e2e.config.BrowserType;
import com.tasks.app.e2e.pages.DashboardPage;
import com.tasks.app.e2e.pages.LoginPage;
import com.tasks.app.e2e.utils.TestDataFactory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
 
import static org.assertj.core.api.Assertions.assertThat;
 
/**
 * Caso 1 (TC-AUTH-001) — RF-01.1, RF-01.2, RF-01.3.
 */
class AuthFlowTest extends BaseE2ETest {
 
    @ParameterizedTest(name = "Registro + login + logout en {0}")
    @EnumSource(BrowserType.class)
    void shouldRegisterLoginAndLogout(BrowserType browser) {
        setupDriver(browser);
 
        String username = TestDataFactory.uniqueUsername();
        String email = TestDataFactory.emailFor(username);
        String password = TestDataFactory.DEFAULT_PASSWORD;
 
        // 1) Registro -> regresa a Login
        LoginPage login = new LoginPage(driver, wait, baseUrl).open()
                .goToRegister()
                .register(username, email, password);
 
        // 2) Login -> Dashboard
        DashboardPage dashboard = login.login(username, password);
        assertThat(dashboard.getDisplayedUsername()).contains(username);
 
        // 3) Logout -> Login
        LoginPage back = dashboard.logout();
        assertThat(driver.getCurrentUrl()).contains("index.html");
        // El token debe haberse limpiado:
        Object token = ((org.openqa.selenium.JavascriptExecutor) driver)
                .executeScript("return window.localStorage.getItem('token');");
        assertThat(token).isNull();
        // Saneamiento: la página de login está disponible.
        assertThat(back).isNotNull();
    }
}