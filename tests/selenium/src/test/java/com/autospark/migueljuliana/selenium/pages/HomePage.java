package com.autospark.migueljuliana.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.FindBy;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class HomePage extends BasePage {

    @FindBy(css = ".logo")
    private WebElement logo;

    @FindBy(css = ".profile-btn")
    private WebElement profileButton;

    @FindBy(css = ".menu-btn")
    private WebElement menuButton;

    public HomePage(WebDriver driver) {
        super(driver);
    }

    /**
     * Navega a la página principal
     */
    public void navigateTo() {
        driver.get("http://autospark_frontend:4200");
    }

    /**
     * Obtiene el título de la página
     */
    public String getPageTitle() {
        return driver.getTitle();
    }

    /**
     * Verifica si el logo está visible
     */
    public boolean isLogoDisplayed() {
        try {
            return logo.isDisplayed();
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Verifica si hay servicios visibles en la página principal
     */
    public boolean areServicesVisible() {
        try {
            return !driver.findElements(By.cssSelector(".servicio")).isEmpty();
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Navega a la página de login
     */
    public LoginPage goToLogin() {

        WebDriverWait wait =
                new WebDriverWait(driver, Duration.ofSeconds(10));

        wait.until(
                ExpectedConditions.elementToBeClickable(profileButton)
        );

        profileButton.click();

        return new LoginPage(driver);
    }

    /**
     * Navega a la página de registro
     */
    public RegisterPage goToRegister() {

        driver.get("http://autospark_frontend:4200/register");

        return new RegisterPage(driver);
    }

    /**
     * Navega a la página de reservas
     */
    public ReservationPage goToReservations() {

        WebDriverWait wait =
                new WebDriverWait(driver, Duration.ofSeconds(10));

        WebElement navigationMenuButton = wait.until(
                ExpectedConditions.elementToBeClickable(
                        By.cssSelector(
                                ".dropdown-toggle, .menu-btn, .profile-btn"
                        )
                )
        );

        navigationMenuButton.click();

        WebElement reservarLink = wait.until(
                ExpectedConditions.elementToBeClickable(
                        By.xpath("//a[contains(text(), 'Reservar')]")
                )
        );

        reservarLink.click();

        return new ReservationPage(driver);
    }

    /**
     * Cierra modal si está visible
     */
    public void closeModalIfPresent() {

        try {

            WebDriverWait wait =
                    new WebDriverWait(driver, Duration.ofSeconds(5));

            WebElement closeButton = wait.until(
                    ExpectedConditions.elementToBeClickable(
                            By.cssSelector(
                                    ".modal .btn-close, .modal button"
                            )
                    )
            );

            closeButton.click();

            wait.until(
                    ExpectedConditions.invisibilityOfElementLocated(
                            By.cssSelector(".modal")
                    )
            );

            System.out.println("Modal cerrado correctamente");

        } catch (Exception e) {

            System.out.println("No había modal visible");
        }
    }

    
}