package com.medicita.selenium.tests;

import com.medicita.selenium.base.BaseTest;
import com.medicita.selenium.pages.LoginPage;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

import static org.assertj.core.api.Assertions.assertThat;

/*
 * TC01–TC03: Pruebas funcionales de la página de login.
 * Cubren: carga de página, toggle de contraseña y navegación hacia registro.
 * No requieren usuario en BD — prueban solo la capa de presentación.
 *
 * Prerequisito: la aplicación debe estar corriendo en BASE_URL (http://localhost).
 * Arrancar con: docker compose -f docker/docker-compose.yml up -d
 */
@DisplayName("TC01-TC03 — Página de login")
class LoginTest extends BaseTest {

    private static final String LOGIN_URL =
            BASE_URL + "/pages/auth/login.html";

    private LoginPage loginPage;

    @BeforeEach
    void openLoginPage() {
        driver.get(LOGIN_URL);
        loginPage = new LoginPage(driver);
    }

    // ── TC01 ──────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("TC01 — La página de login carga con el título correcto")
    void paginaLogin_carga_tituloEsCorrecto() {
        // La página debe cargar y mostrar el título configurado en el <head>
        assertThat(loginPage.getPageTitle())
                .isEqualTo("MediCita – Iniciar sesión");
    }

    // ── TC02 ──────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("TC02 — El botón de ojo alterna la visibilidad de la contraseña")
    void paginaLogin_togglePassword_cambiaElTipoDelInput() {
        // El input inicia como type="password" (texto oculto)
        assertThat(loginPage.getPasswordInputType()).isEqualTo("password");
        assertThat(loginPage.getEyeIconClass()).contains("bi-eye");

        // Al hacer clic debe cambiar a type="text" (texto visible)
        loginPage.clickTogglePassword();
        assertThat(loginPage.getPasswordInputType()).isEqualTo("text");
        assertThat(loginPage.getEyeIconClass()).contains("bi-eye-slash");

        // Un segundo clic debe restaurar el estado original
        loginPage.clickTogglePassword();
        assertThat(loginPage.getPasswordInputType()).isEqualTo("password");
    }

    // ── TC03 ──────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("TC03 — El enlace 'Regístrate aquí' navega a la página de registro")
    void paginaLogin_enlaceRegistro_navegaAPaginaRegistro() {
        // El enlace de registro debe estar visible en la página
        assertThat(loginPage.isRegisterLinkVisible()).isTrue();

        // Al hacer clic debe navegar a register.html
        loginPage.clickRegisterLink();

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));
        wait.until(ExpectedConditions.titleContains("Registro"));

        assertThat(driver.getTitle()).isEqualTo("MediCita – Registro");
        assertThat(driver.getCurrentUrl()).contains("register.html");
    }
}
