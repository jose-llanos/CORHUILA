package com.autospark.migueljuliana.selenium.tests;

import com.autospark.migueljuliana.selenium.pages.HomePage;
import com.autospark.migueljuliana.selenium.pages.LoginPage;
import com.aventstack.extentreports.Status;
import org.openqa.selenium.By;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.time.Duration;

public class LoginTest extends BaseTest {

    /**
     * TC-FUNC-002: Login exitoso con credenciales correctas
     */
    @Test(description = "TC-FUNC-002: Login exitoso con credenciales correctas")
    public void testSuccessfulLogin() {
        test = extent.createTest("Login exitoso");

        LoginPage loginPage = new LoginPage(driver);
        loginPage.navigateTo();
        test.log(Status.INFO, "Página de login cargada");

        // ✅ CAMBIAR: Usar loginSuccess en lugar de login
        HomePage homePage = loginPage.loginSuccess("juan@test.com", "Password123");
        test.log(Status.INFO, "Credenciales ingresadas");

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));
        wait.until(driver -> homePage.isLogoDisplayed());

        Assert.assertTrue(homePage.isLogoDisplayed(),
                "El logo debería ser visible después del login");

        test.log(Status.PASS, "Login exitoso - Redirigido a página principal");
    }

    /**
     * TC-FUNC-003: Login fallido con credenciales incorrectas
     */
    @Test(description = "TC-FUNC-003: Login fallido con credenciales incorrectas")
    public void testFailedLogin() {
        test = extent.createTest("Login fallido con credenciales incorrectas");

        LoginPage loginPage = new LoginPage(driver);
        loginPage.navigateTo();
        test.log(Status.INFO, "Página de login cargada");

        // ✅ CAMBIAR: Usar loginFail en lugar de login
        loginPage.loginFail("usuario_no_existe@test.com", "WrongPassword123");
        test.log(Status.INFO, "Credenciales incorrectas ingresadas");

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(5));

        // Esperar que aparezca el modal de error
        wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector(".modal-content")));

        // Verificar que el modal de error está visible
        WebElement errorModal = driver.findElement(By.cssSelector(".modal-content"));
        Assert.assertTrue(errorModal.isDisplayed(), "El modal de error debería mostrarse");

        // Verificar que la URL sigue siendo /login
        String currentUrl = driver.getCurrentUrl();
        Assert.assertTrue(currentUrl.contains("/login"),
                "Debe permanecer en la página de login después de credenciales incorrectas");

        test.log(Status.PASS, "Login fallido correctamente - Permanece en página de login");
    }
}