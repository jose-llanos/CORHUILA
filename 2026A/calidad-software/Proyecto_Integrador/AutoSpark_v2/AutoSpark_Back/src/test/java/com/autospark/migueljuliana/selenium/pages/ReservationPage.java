package com.autospark.migueljuliana.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.FindBy;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.Select;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.List;

public class ReservationPage extends BasePage {

    @FindBy(name = "vehicleType")
    private WebElement vehicleTypeSelect;

    @FindBy(name = "licensePlate")
    private WebElement licensePlateInput;

    @FindBy(name = "serviceType")
    private WebElement serviceTypeSelect;

    @FindBy(name = "value")
    private WebElement valueInput;

    @FindBy(name = "reservationDate")
    private WebElement dateInput;

    @FindBy(name = "reservationTime")
    private WebElement timeSelect;

    @FindBy(css = "button[type='submit']")
    private WebElement submitButton;

    @FindBy(css = ".modal-content")
    private WebElement successModal;

    public ReservationPage(WebDriver driver) {
        super(driver);
    }

    public void navigateTo() {
        driver.get("http://localhost:4200/reserves");
        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));
        wait.until(ExpectedConditions.visibilityOfElementLocated(By.name("vehicleType")));
    }

    /**
     * Crear reserva con fecha automática (mañana) y hora específica
     */
    /**
     * Crear reserva con fecha y hora específicas
     */
    public void createReservation(String vehicleType, String licensePlate,
                                  String serviceType, String value,
                                  String date, String hour) {

        createReservationWithDateAndTime(
                vehicleType,
                licensePlate,
                serviceType,
                value,
                date,
                hour
        );
    }

    /**
     * Crear reserva con fecha y hora personalizadas
     */
    public void createReservationWithDateAndTime(String vehicleType,
                                                 String licensePlate,
                                                 String serviceType,
                                                 String value,
                                                 String date,
                                                 String hour) {
        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));

        // Seleccionar tipo de vehículo
        wait.until(ExpectedConditions.presenceOfElementLocated(By.name("vehicleType")));
        Select vehicleSelect = new Select(driver.findElement(By.name("vehicleType")));
        vehicleSelect.selectByValue(vehicleType);
        System.out.println("Vehículo seleccionado: " + vehicleSelect.getFirstSelectedOption().getText());

        // Placa
        driver.findElement(By.name("licensePlate")).clear();
        driver.findElement(By.name("licensePlate")).sendKeys(licensePlate);

        // Servicio
        WebDriverWait longWait = new WebDriverWait(driver, Duration.ofSeconds(15));
        longWait.until(ExpectedConditions.presenceOfElementLocated(By.name("serviceType")));

        Select serviceSelect = new Select(driver.findElement(By.name("serviceType")));
        serviceSelect.selectByVisibleText(serviceType);
        System.out.println("Servicio seleccionado: " + serviceType);

        // Valor
        WebElement valueInput = driver.findElement(By.name("value"));
        valueInput.clear();
        valueInput.sendKeys(value);

        // FECHA Y HORA COMBINADAS en formato "YYYY-MM-DDTHH:MM"
        String dateTime = date + "T" + hour;
        WebElement dateTimeInput = driver.findElement(By.name("reservationDate"));
        dateTimeInput.clear();
        dateTimeInput.sendKeys(dateTime);
        System.out.println("Fecha y hora seleccionada: " + dateTime);

        // Enviar formulario
        WebElement submitButton = driver.findElement(By.cssSelector("button[type='submit']"));
        wait.until(ExpectedConditions.elementToBeClickable(submitButton));
        submitButton.click();

        System.out.println("Formulario de reserva enviado");
    }

    public boolean isSuccessModalDisplayed() {

        try {

            WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));

            wait.until(ExpectedConditions.visibilityOf(successModal));

            return successModal.isDisplayed();

        } catch (Exception e) {

            System.out.println(
                    "Modal de éxito no encontrado: "
                            + e.getMessage()
            );

            return false;
        }
    }

    public String getSuccessMessage() {
        return successModal.getText();
    }

    public String getErrorMessage() {

        try {

            WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(5));

            wait.until(ExpectedConditions.visibilityOf(successModal));

            return successModal.getText();

        } catch (Exception e) {

            return "";
        }
    }

    public void closeModal() {

        try {

            WebElement closeButton =
                    driver.findElement(By.cssSelector(".modal button"));

            closeButton.click();

        } catch (Exception e) {

            System.out.println("Modal ya estaba cerrado");
        }
    }
}