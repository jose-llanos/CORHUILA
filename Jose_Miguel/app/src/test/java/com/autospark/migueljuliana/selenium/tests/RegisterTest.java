package com.autospark.migueljuliana.selenium.tests;

import com.autospark.migueljuliana.selenium.pages.HomePage;
import com.autospark.migueljuliana.selenium.pages.RegisterPage;
import com.aventstack.extentreports.Status;

import org.openqa.selenium.support.ui.WebDriverWait;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.time.Duration;

public class RegisterTest extends BaseTest {

    /**
     * TC-FUNC-001: Registro exitoso de nuevo usuario
     * Flujo: Navegar a registro -> Completar formulario -> Verificar modal de éxito
     */
    @Test(description = "TC-FUNC-001: Registro exitoso de nuevo usuario")
    public void testSuccessfulRegistration() {

        test = extent.createTest("Registro exitoso de usuario");

        HomePage homePage = new HomePage(driver);

        homePage.navigateTo();

        test.log(Status.INFO, "Página de inicio cargada");

        RegisterPage registerPage = homePage.goToRegister();

        test.log(Status.INFO, "Navegando a página de registro");

        String timestamp = String.valueOf(System.currentTimeMillis());

        String email = "test_" + timestamp + "@test.com";

        registerPage.register(
                "Test User",
                email,
                "Password123",
                "12345678",
                "3001234567",
                "ABC123"
        );

        test.log(
                Status.INFO,
                "Formulario de registro completado con email: " + email
        );

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));

        wait.until(driver ->
                registerPage.isSuccessModalDisplayed()
        );

        Assert.assertTrue(
                registerPage.isSuccessModalDisplayed(),
                "El modal de éxito debería mostrarse"
        );

        test.log(
                Status.PASS,
                "Modal de éxito visible - Registro exitoso"
        );
    }
}