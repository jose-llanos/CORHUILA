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
    public void createReservation(String vehicleType, String licensePlate,
                                  String serviceType, String value, String hour) {

        LocalDate tomorrow = LocalDate.now().plusDays(1);

        String date = tomorrow.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));

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

        wait.until(ExpectedConditions.presenceOfElementLocated(By.name("vehicleType")));

        Select vehicleSelect = new Select(vehicleTypeSelect);

        vehicleSelect.selectByValue(vehicleType);

        System.out.println(
                "Vehículo seleccionado: "
                        + vehicleSelect.getFirstSelectedOption().getText()
        );

        licensePlateInput.clear();

        licensePlateInput.sendKeys(licensePlate);

        WebDriverWait longWait = new WebDriverWait(driver, Duration.ofSeconds(15));

        longWait.until(ExpectedConditions.presenceOfElementLocated(By.name("serviceType")));

        longWait.until(driver -> {
            Select select = new Select(serviceTypeSelect);
            return select.getOptions().size() > 1;
        });

        Select serviceSelect = new Select(serviceTypeSelect);

        List<WebElement> options = serviceSelect.getOptions();

        System.out.println("Opciones de servicio disponibles (" + options.size() + "):");

        for (WebElement opt : options) {
            System.out.println(" - '" + opt.getText() + "'");
        }

        if (options.size() > 1) {

            boolean found = false;

            for (int i = 0; i < options.size(); i++) {

                if (options.get(i).getText().equalsIgnoreCase(serviceType)) {

                    serviceSelect.selectByIndex(i);

                    found = true;

                    break;
                }
            }

            if (!found) {
                serviceSelect.selectByIndex(1);
            }

            System.out.println(
                    "Servicio seleccionado: "
                            + serviceSelect.getFirstSelectedOption().getText()
            );

        } else {

            throw new IllegalStateException(
                    "No hay servicios disponibles para seleccionar"
            );
        }

        wait.until(ExpectedConditions.attributeToBeNotEmpty(valueInput, "value"));

        valueInput.clear();

        valueInput.sendKeys(value);

        wait.until(ExpectedConditions.visibilityOf(dateInput));

        dateInput.clear();

        dateInput.sendKeys(date);

        System.out.println("Fecha seleccionada: " + date);

        wait.until(ExpectedConditions.visibilityOf(timeSelect));

        Select timeSelectObj = new Select(timeSelect);

        timeSelectObj.selectByValue(hour);

        System.out.println("Hora seleccionada: " + hour);

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