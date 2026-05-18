package com.medicita.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

/*
 * Page Object para la página de login (/pages/auth/login.html).
 * Encapsula todos los locators y acciones de esa página.
 * Los tests nunca tocan el DOM directamente — lo hacen a través de este objeto.
 */
public class LoginPage {

    private final WebDriver driver;
    private final WebDriverWait wait;

    // ── Locators (IDs sacados del HTML real) ──────────────────────────────────
    private static final By EMAIL_INPUT      = By.id("email");
    private static final By PASSWORD_INPUT   = By.id("password");
    private static final By SUBMIT_BUTTON    = By.id("btn-submit");
    private static final By TOGGLE_PASSWORD  = By.id("toggle-password");
    private static final By EYE_ICON        = By.id("eye-icon");
    private static final By REGISTER_LINK   = By.linkText("Regístrate aquí");

    public LoginPage(WebDriver driver) {
        this.driver = driver;
        this.wait   = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    // ── Acciones ──────────────────────────────────────────────────────────────

    public void enterEmail(String email) {
        WebElement el = wait.until(ExpectedConditions.visibilityOfElementLocated(EMAIL_INPUT));
        el.clear();
        el.sendKeys(email);
    }

    public void enterPassword(String password) {
        WebElement el = driver.findElement(PASSWORD_INPUT);
        el.clear();
        el.sendKeys(password);
    }

    public void clickSubmit() {
        driver.findElement(SUBMIT_BUTTON).click();
    }

    public void clickTogglePassword() {
        driver.findElement(TOGGLE_PASSWORD).click();
    }

    public void clickRegisterLink() {
        driver.findElement(REGISTER_LINK).click();
    }

    // ── Consultas (getters) ───────────────────────────────────────────────────

    public String getPageTitle() {
        return driver.getTitle();
    }

    public String getCurrentUrl() {
        return driver.getCurrentUrl();
    }

    public String getPasswordInputType() {
        return driver.findElement(PASSWORD_INPUT).getAttribute("type");
    }

    public String getEyeIconClass() {
        return driver.findElement(EYE_ICON).getAttribute("class");
    }

    public boolean isSubmitEnabled() {
        return driver.findElement(SUBMIT_BUTTON).isEnabled();
    }

    public boolean isRegisterLinkVisible() {
        return driver.findElement(REGISTER_LINK).isDisplayed();
    }
}
