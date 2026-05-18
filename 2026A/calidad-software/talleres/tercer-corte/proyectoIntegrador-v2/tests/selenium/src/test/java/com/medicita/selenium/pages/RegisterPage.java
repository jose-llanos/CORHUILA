package com.medicita.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

/*
 * Page Object para la página de registro (/pages/auth/register.html).
 * La validación del lado del cliente (JavaScript) añade la clase CSS
 * "is-invalid" a los campos vacíos cuando se intenta enviar el formulario.
 * Eso es lo que verificamos en los tests de validación.
 */
public class RegisterPage {

    private final WebDriver driver;
    private final WebDriverWait wait;

    // ── Locators ──────────────────────────────────────────────────────────────
    private static final By FIRST_NAME_INPUT  = By.id("firstName");
    private static final By LAST_NAME_INPUT   = By.id("lastName");
    private static final By EMAIL_INPUT       = By.id("email");
    private static final By PASSWORD_INPUT    = By.id("password");
    private static final By DOCUMENT_INPUT    = By.id("documentNumber");
    private static final By PHONE_INPUT       = By.id("phone");
    private static final By BIRTH_DATE_INPUT  = By.id("birthDate");
    private static final By SUBMIT_BUTTON     = By.id("btn-submit");
    private static final By LOGIN_LINK        = By.linkText("Inicia sesión");

    public RegisterPage(WebDriver driver) {
        this.driver = driver;
        this.wait   = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    // ── Acciones ──────────────────────────────────────────────────────────────

    public void waitUntilLoaded() {
        wait.until(ExpectedConditions.visibilityOfElementLocated(FIRST_NAME_INPUT));
    }

    public void enterFirstName(String value) {
        typeInField(FIRST_NAME_INPUT, value);
    }

    public void enterLastName(String value) {
        typeInField(LAST_NAME_INPUT, value);
    }

    public void enterEmail(String value) {
        typeInField(EMAIL_INPUT, value);
    }

    public void enterPassword(String value) {
        typeInField(PASSWORD_INPUT, value);
    }

    public void enterDocumentNumber(String value) {
        typeInField(DOCUMENT_INPUT, value);
    }

    public void enterPhone(String value) {
        typeInField(PHONE_INPUT, value);
    }

    public void enterBirthDate(String value) {
        typeInField(BIRTH_DATE_INPUT, value);
    }

    public void clickSubmit() {
        driver.findElement(SUBMIT_BUTTON).click();
    }

    public void clickLoginLink() {
        // Scroll hasta el enlace antes de hacer clic — evita ElementClickIntercepted
        // cuando el footer o un toast lo tapan en ventana pequeña
        WebElement link = driver.findElement(LOGIN_LINK);
        ((JavascriptExecutor) driver).executeScript("arguments[0].scrollIntoView(true);", link);
        link.click();
    }

    // ── Consultas ─────────────────────────────────────────────────────────────

    public String getPageTitle() {
        return driver.getTitle();
    }

    public String getCurrentUrl() {
        return driver.getCurrentUrl();
    }

    public boolean isFirstNameVisible() {
        return driver.findElement(FIRST_NAME_INPUT).isDisplayed();
    }

    public boolean isDocumentInputVisible() {
        return driver.findElement(DOCUMENT_INPUT).isDisplayed();
    }

    public boolean isSubmitVisible() {
        return driver.findElement(SUBMIT_BUTTON).isDisplayed();
    }

    // Verifica si el JS de validación marcó el campo como inválido (clase is-invalid)
    public boolean isFirstNameInvalid() {
        return hasClass(FIRST_NAME_INPUT, "is-invalid");
    }

    public boolean isLastNameInvalid() {
        return hasClass(LAST_NAME_INPUT, "is-invalid");
    }

    public boolean isDocumentInvalid() {
        return hasClass(DOCUMENT_INPUT, "is-invalid");
    }

    // ── Utilidades privadas ───────────────────────────────────────────────────

    private void typeInField(By locator, String value) {
        WebElement el = driver.findElement(locator);
        el.clear();
        el.sendKeys(value);
    }

    private boolean hasClass(By locator, String cssClass) {
        String classes = driver.findElement(locator).getAttribute("class");
        return classes != null && classes.contains(cssClass);
    }
}
