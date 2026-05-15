package com.autospark.migueljuliana.selenium.tests;

import com.autospark.migueljuliana.selenium.pages.HomePage;
import com.autospark.migueljuliana.selenium.pages.LoginPage;
import com.autospark.migueljuliana.selenium.pages.ReservationPage;
import com.aventstack.extentreports.Status;

import org.openqa.selenium.By;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.time.Duration;

public class ReservationTest extends BaseTest {

    /**
     * TC-FUNC-005: Crear reserva exitosamente (requiere login)
     */
    @Test(description = "TC-FUNC-005: Crear reserva exitosamente")
    public void testCreateReservation() {
        test = extent.createTest("Crear reserva exitosamente");

        LoginPage loginPage = new LoginPage(driver);
        loginPage.navigateTo();
        test.log(Status.INFO, "Página de login cargada");

        // Login exitoso y obtener HomePage
        HomePage homePage = loginPage.loginSuccess("juan@test.com", "Password123");
        test.log(Status.INFO, "Login completado");

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(20));

        // ESPERAR a que la URL ya no sea /login (el login fue exitoso)
        wait.until(ExpectedConditions.not(ExpectedConditions.urlContains("login")));

        // Navegar a la página de reservas usando el menú
        ReservationPage reservationPage = homePage.goToReservations();
        test.log(Status.INFO, "Página de reservas cargada");

        // Ahora sí, esperar el formulario de reservas
        wait.until(ExpectedConditions.visibilityOfElementLocated(By.name("vehicleType")));

        String fecha = "2026-05-20";
        String hora = "15:00:00";

        reservationPage.createReservation(
                "CARRO",
                "ABC123",
                "Lavado Basico",
                "25000",
                fecha,
                hora
        );

        test.log(Status.INFO, "Formulario de reserva completado");

        wait.until(driver -> reservationPage.isSuccessModalDisplayed());

        Assert.assertTrue(reservationPage.isSuccessModalDisplayed(),
                "El modal de éxito debería mostrarse");

        test.log(Status.PASS, "Reserva creada exitosamente");
    }

    /**
     * TC-FUNC-006: Crear reserva con fecha específica
     */
    @Test(description = "TC-FUNC-006: Crear reserva con fecha específica")
    public void testCreateReservationWithSpecificDate() {
        test = extent.createTest("Crear reserva con fecha específica");

        LoginPage loginPage = new LoginPage(driver);
        loginPage.navigateTo();

        HomePage homePage = loginPage.loginSuccess("juan@test.com", "Password123");
        test.log(Status.INFO, "Login completado");

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(20));

        // ESPERAR a que la URL ya no sea /login (el login fue exitoso)
        wait.until(ExpectedConditions.not(ExpectedConditions.urlContains("login")));

        // Navegar a la página de reservas usando el menú
        ReservationPage reservationPage = homePage.goToReservations();
        test.log(Status.INFO, "Página de reservas cargada");

        wait.until(ExpectedConditions.visibilityOfElementLocated(By.name("vehicleType")));

        String fecha = "2026-05-20";
        String hora = "14:00:00";

        // Cambiar el texto del servicio para que coincida con el frontend
        reservationPage.createReservationWithDateAndTime(
                "CAMIONETA",
                "XYZ789",
                "Lavado Premium",
                "45000",
                fecha,
                hora
        );

        test.log(Status.INFO, "Reserva creada para fecha: " + fecha + " - Hora: " + hora);

        wait.until(driver -> reservationPage.isSuccessModalDisplayed());

        Assert.assertTrue(reservationPage.isSuccessModalDisplayed(),
                "El modal de éxito debería mostrarse");

        test.log(Status.PASS, "Reserva con fecha específica creada exitosamente");
    }
}