package com.medicita.selenium.tests;

import com.medicita.selenium.base.BaseTest;
import com.medicita.selenium.pages.RegisterPage;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

import static org.assertj.core.api.Assertions.assertThat;

/*
 * TC04–TC06: Pruebas funcionales de la página de registro.
 * Cubren: carga de página, presencia de campos y validación JavaScript
 * del lado del cliente (la función setFieldError añade clase "is-invalid").
 *
 * Prerequisito: la aplicación debe estar corriendo en BASE_URL (http://localhost).
 * Arrancar con: docker compose -f docker/docker-compose.yml up -d
 */
@DisplayName("TC04-TC06 — Página de registro")
class RegisterTest extends BaseTest {

    private static final String REGISTER_URL =
            BASE_URL + "/pages/auth/register.html";

    private RegisterPage registerPage;

    @BeforeEach
    void openRegisterPage() {
        driver.get(REGISTER_URL);
        registerPage = new RegisterPage(driver);
        registerPage.waitUntilLoaded();
    }

    // ── TC04 ──────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("TC04 — La página de registro carga con el título correcto")
    void paginaRegistro_carga_tituloEsCorrecto() {
        assertThat(registerPage.getPageTitle())
                .isEqualTo("MediCita – Registro");
    }

    // ── TC05 ──────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("TC05 — El formulario muestra todos los campos obligatorios")
    void paginaRegistro_carga_todosLosCamposObligatoriosVisibles() {
        // Los tres campos más críticos deben estar en pantalla
        assertThat(registerPage.isFirstNameVisible())
                .as("Campo 'Nombre' debe ser visible")
                .isTrue();

        assertThat(registerPage.isDocumentInputVisible())
                .as("Campo 'Número de documento' debe ser visible")
                .isTrue();

        assertThat(registerPage.isSubmitVisible())
                .as("Botón 'Crear cuenta' debe ser visible")
                .isTrue();
    }

    // ── TC06 ──────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("TC06 — Enviar el formulario vacío marca los campos obligatorios como inválidos")
    void paginaRegistro_submitSinDatos_marcaCamposObligatoriosComoInvalidos() {
        // Sin llenar ningún campo, el JS de validación debe marcar los campos requeridos
        registerPage.clickSubmit();

        // Pequeña espera para que el JS aplique las clases CSS
        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(3));
        wait.until(d -> registerPage.isFirstNameInvalid());

        assertThat(registerPage.isFirstNameInvalid())
                .as("'Nombre' debe quedar marcado como inválido")
                .isTrue();

        assertThat(registerPage.isLastNameInvalid())
                .as("'Apellido' debe quedar marcado como inválido")
                .isTrue();

        assertThat(registerPage.isDocumentInvalid())
                .as("'Número de documento' debe quedar marcado como inválido")
                .isTrue();
    }

    // ── TC07 (bonus) ──────────────────────────────────────────────────────────

    @Test
    @DisplayName("TC07 — El enlace 'Inicia sesión' navega de vuelta a login")
    void paginaRegistro_enlaceLogin_navegaAPaginaLogin() {
        registerPage.clickLoginLink();

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));
        wait.until(ExpectedConditions.titleContains("Iniciar"));

        assertThat(driver.getTitle()).isEqualTo("MediCita – Iniciar sesión");
        assertThat(driver.getCurrentUrl()).contains("login.html");
    }
}
