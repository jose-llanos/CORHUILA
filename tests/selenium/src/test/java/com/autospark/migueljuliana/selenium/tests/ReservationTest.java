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
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

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

        HomePage homePage = loginPage.loginSuccess("juan@test.com", "Password123");
        test.log(Status.INFO, "Login completado");

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(30));

        wait.until(ExpectedConditions.or(
                ExpectedConditions.visibilityOfElementLocated(By.cssSelector(".logo")),
                ExpectedConditions.visibilityOfElementLocated(By.cssSelector(".profile-btn")),
                ExpectedConditions.visibilityOfElementLocated(By.cssSelector("app-header"))
        ));

        ReservationPage reservationPage = homePage.goToReservations();
        test.log(Status.INFO, "Página de reservas cargada");

        wait.until(ExpectedConditions.visibilityOfElementLocated(By.name("vehicleType")));

        LocalDate futureDate = LocalDate.now().plusDays(3);
        String fecha = futureDate.format(DateTimeFormatter.ISO_LOCAL_DATE);
        String hora = "15:00:00";

        System.out.println("Usando fecha: " + fecha + " y hora: " + hora);

        reservationPage.createReservationWithDateAndTime(
                "CARRO",
                "ABC123",
                "Lavado Basico",
                "25000",
                fecha,
                hora
        );

        test.log(Status.INFO, "Formulario de reserva completado con fecha: " + fecha + " y hora: " + hora);

        wait.until(driver -> reservationPage.isSuccessModalDisplayed());

        Assert.assertTrue(
                reservationPage.isSuccessModalDisplayed(),
                "El modal de éxito debería mostrarse"
        );

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
        test.log(Status.INFO, "Página de login cargada");

        HomePage homePage = loginPage.loginSuccess("juan@test.com", "Password123");
        test.log(Status.INFO, "Login completado");

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(30));

        wait.until(ExpectedConditions.or(
                ExpectedConditions.visibilityOfElementLocated(By.cssSelector(".logo")),
                ExpectedConditions.visibilityOfElementLocated(By.cssSelector(".profile-btn")),
                ExpectedConditions.visibilityOfElementLocated(By.cssSelector("app-header"))
        ));

        ReservationPage reservationPage = homePage.goToReservations();
        test.log(Status.INFO, "Página de reservas cargada");

        wait.until(ExpectedConditions.visibilityOfElementLocated(By.name("vehicleType")));

        LocalDate specificFutureDate = LocalDate.now().plusDays(3);
        String fecha = specificFutureDate.format(DateTimeFormatter.ISO_LOCAL_DATE);
        String hora = "14:00:00";

        System.out.println("Usando fecha específica: " + fecha + " y hora: " + hora);

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

        Assert.assertTrue(
                reservationPage.isSuccessModalDisplayed(),
                "El modal de éxito debería mostrarse"
        );

        test.log(Status.PASS, "Reserva con fecha específica creada exitosamente");
    }
}