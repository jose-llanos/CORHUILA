package com.autospark.migueljuliana.selenium.tests;

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

        loginPage.login("juan@test.com", "Password123");

        test.log(Status.INFO, "Login completado");

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));

        wait.until(ExpectedConditions.urlContains("home"));

        ReservationPage reservationPage = new ReservationPage(driver);

        reservationPage.navigateTo();

        test.log(Status.INFO, "Página de reservas cargada");

        wait.until(
                ExpectedConditions.visibilityOfElementLocated(
                        By.name("vehicleType")
                )
        );

        reservationPage.createReservation(
                "CARRO",
                "ABC123",
                "LAVADO BASICO",
                "25000",
                "10:00:00"
        );

        test.log(
                Status.INFO,
                "Formulario de reserva completado con fecha y hora"
        );

        wait.until(driver ->
                reservationPage.isSuccessModalDisplayed()
        );

        Assert.assertTrue(
                reservationPage.isSuccessModalDisplayed(),
                "El modal de éxito debería mostrarse"
        );

        test.log(
                Status.PASS,
                "Modal de éxito visible - Reserva creada exitosamente"
        );
    }

    /**
     * TC-FUNC-006: Crear reserva con fecha específica
     */
    @Test(description = "TC-FUNC-006: Crear reserva con fecha específica")
    public void testCreateReservationWithSpecificDate() {

        test = extent.createTest("Crear reserva con fecha específica");

        LoginPage loginPage = new LoginPage(driver);

        loginPage.navigateTo();

        loginPage.login("juan@test.com", "Password123");

        test.log(Status.INFO, "Login completado");

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));

        wait.until(ExpectedConditions.urlContains("home"));

        ReservationPage reservationPage = new ReservationPage(driver);

        reservationPage.navigateTo();

        test.log(Status.INFO, "Página de reservas cargada");

        wait.until(
                ExpectedConditions.visibilityOfElementLocated(
                        By.name("vehicleType")
                )
        );

        String fecha = "2026-05-20";

        String hora = "14:00:00";

        reservationPage.createReservationWithDateAndTime(
                "CAMIONETA",
                "XYZ789",
                "LAVADO PREMIUM",
                "45000",
                fecha,
                hora
        );

        test.log(
                Status.INFO,
                "Reserva creada para fecha: " + fecha + " - Hora: " + hora
        );

        wait.until(driver ->
                reservationPage.isSuccessModalDisplayed()
        );

        Assert.assertTrue(
                reservationPage.isSuccessModalDisplayed(),
                "El modal de éxito debería mostrarse"
        );

        test.log(
                Status.PASS,
                "Reserva con fecha específica creada exitosamente"
        );
    }
}