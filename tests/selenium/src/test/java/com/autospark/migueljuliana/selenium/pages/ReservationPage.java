package com.autospark.migueljuliana.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.FindBy;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.Select;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

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
        driver.get("http://autospark_frontend:4200/reserves");

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));

        wait.until(
                ExpectedConditions.visibilityOfElementLocated(
                        By.name("vehicleType")
                )
        );
    }

    /**
     * Crear reserva con fecha y hora personalizadas
     * Ahora maneja fecha y hora por separado como en el frontend
     */
    public void createReservationWithDateAndTime(String vehicleType,
                                                 String licensePlate,
                                                 String serviceType,
                                                 String value,
                                                 String date,
                                                 String hour) {

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));

        wait.until(
                ExpectedConditions.presenceOfElementLocated(
                        By.name("vehicleType")
                )
        );

        // Seleccionar tipo de vehículo
        Select vehicleSelect =
                new Select(driver.findElement(By.name("vehicleType")));

        vehicleSelect.selectByValue(vehicleType);

        System.out.println(
                "Vehículo seleccionado: "
                        + vehicleSelect.getFirstSelectedOption().getText()
        );

        // Ingresar placa
        WebElement licensePlateElement =
                driver.findElement(By.name("licensePlate"));

        licensePlateElement.clear();
        licensePlateElement.sendKeys(licensePlate);

        // Seleccionar tipo de servicio
        WebDriverWait longWait =
                new WebDriverWait(driver, Duration.ofSeconds(15));

        longWait.until(
                ExpectedConditions.presenceOfElementLocated(
                        By.name("serviceType")
                )
        );

        Select serviceSelect =
                new Select(driver.findElement(By.name("serviceType")));

        serviceSelect.selectByVisibleText(serviceType);

        System.out.println("Servicio seleccionado: " + serviceType);

        // Ingresar valor
        WebElement saleValueInput =
                driver.findElement(By.name("value"));

        saleValueInput.clear();
        saleValueInput.sendKeys(value);

        // Ingresar fecha (formato YYYY-MM-DD)
        WebElement reservationDateInput =
                driver.findElement(By.name("reservationDate"));

        reservationDateInput.clear();
        reservationDateInput.sendKeys(date);

        System.out.println("Fecha seleccionada: " + date);

        // 👇 CORREGIDO: Usar un nombre diferente para la variable local
        // Seleccionar hora (del dropdown)
        WebElement reservationTimeElement =
                driver.findElement(By.name("reservationTime"));

        Select horaSelect = new Select(reservationTimeElement);  // Renombrado a 'horaSelect'
        horaSelect.selectByValue(hour);

        System.out.println("Hora seleccionada: " + hour);

        // Hacer clic en el botón de submit
        WebElement reservationSubmitButton =
                driver.findElement(By.cssSelector("button[type='submit']"));

        wait.until(
                ExpectedConditions.elementToBeClickable(
                        reservationSubmitButton
                )
        );

        reservationSubmitButton.click();

        System.out.println("Formulario de reserva enviado");
    }

    public boolean isSuccessModalDisplayed() {

        try {

            WebDriverWait wait =
                    new WebDriverWait(driver, Duration.ofSeconds(10));

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

            WebDriverWait wait =
                    new WebDriverWait(driver, Duration.ofSeconds(5));

            wait.until(ExpectedConditions.visibilityOf(successModal));

            return successModal.getText();

        } catch (Exception e) {

            return "";
        }
    }

        public void closeModal() {

                try {

                        WebDriverWait wait =
                                new WebDriverWait(driver, Duration.ofSeconds(10));

                if (!driver.findElements(By.cssSelector(".modal")).isEmpty()) {

                        WebElement closeButton =
                                wait.until(
                                        ExpectedConditions.elementToBeClickable(
                                                By.cssSelector(".modal .btn-close, .modal button")
                                        )
                                );

                        ((JavascriptExecutor) driver).executeScript(
                                "arguments[0].click();",
                                closeButton
                        );

                wait.until(
                        ExpectedConditions.invisibilityOfElementLocated(
                                By.cssSelector(".modal")
                        )
                );

                System.out.println("Modal cerrado correctamente");
        }

        } catch (Exception e) {

                System.out.println(
                        "Modal ya estaba cerrado o no se pudo cerrar: "
                                + e.getMessage()
                );
                }
        }
}



