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

    @FindBy(css = ".servicio")
    private java.util.List<WebElement> serviceCards;

    public HomePage(WebDriver driver) {
        super(driver);
    }

    /**
     * Navega a la página principal
     */
    public void navigateTo() {
        driver.get("http://localhost:4200");
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
            return !serviceCards.isEmpty();
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Navega a la página de login
     */
    public LoginPage goToLogin() {

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));

        wait.until(ExpectedConditions.elementToBeClickable(profileButton));

        profileButton.click();

        return new LoginPage(driver);
    }

    /**
     * Navega a la página de registro
     */
    public RegisterPage goToRegister() {
        driver.get("http://localhost:4200/register");
        return new RegisterPage(driver);
    }

    public ReservationPage goToReservations() {
        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));

        // Primero, abrir el menú desplegable (si existe)
        WebElement menuButton = wait.until(ExpectedConditions.elementToBeClickable(By.cssSelector(".dropdown-toggle, .menu-btn, .profile-btn")));
        menuButton.click();

        // Luego, hacer clic en la opción "Reservar"
        WebElement reservarLink = wait.until(ExpectedConditions.elementToBeClickable(By.xpath("//a[contains(text(), 'Reservar')]")));
        reservarLink.click();

        return new ReservationPage(driver);
    }
}