package com.autospark.migueljuliana.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.FindBy;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class LoginPage extends BasePage {

    private static final String LOGIN_URL = "http://autospark_frontend:4200/login";

    @FindBy(name = "email")
    private WebElement emailInput;

    @FindBy(name = "password")
    private WebElement passwordInput;

    @FindBy(css = "button[type='submit']")
    private WebElement loginButton;

    @FindBy(css = ".modal-content")
    private WebElement errorModal;

    @FindBy(linkText = "Regístrate")
    private WebElement registerLink;

    @FindBy(linkText = "¿Olvidó su contraseña?")
    private WebElement forgotPasswordLink;

    public LoginPage(WebDriver driver) {
        super(driver);
    }

    public void navigateTo() {
        driver.get(LOGIN_URL);

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(20));
        wait.until(ExpectedConditions.visibilityOfElementLocated(By.name("email")));
        wait.until(ExpectedConditions.visibilityOfElementLocated(By.name("password")));
    }

    /**
     * Login exitoso.
     * No depende únicamente de que cambie la URL, porque la app puede autenticar
     * y quedarse momentáneamente en /login o redirigir de forma asíncrona.
     */
    public HomePage loginSuccess(String email, String password) {
        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(30));

        WebElement emailField = wait.until(
                ExpectedConditions.elementToBeClickable(By.name("email"))
        );
        emailField.clear();
        emailField.sendKeys(email);

        WebElement passwordField = wait.until(
                ExpectedConditions.elementToBeClickable(By.name("password"))
        );
        passwordField.clear();
        passwordField.sendKeys(password);

        WebElement loginBtn = wait.until(
                ExpectedConditions.elementToBeClickable(By.cssSelector("button[type='submit']"))
        );
        loginBtn.click();

        wait.until(ExpectedConditions.or(
                ExpectedConditions.not(ExpectedConditions.urlContains("/login")),
                ExpectedConditions.urlContains("/services"),
                ExpectedConditions.visibilityOfElementLocated(By.cssSelector(".logo")),
                ExpectedConditions.visibilityOfElementLocated(By.cssSelector(".profile-btn")),
                ExpectedConditions.visibilityOfElementLocated(By.cssSelector("app-header"))
        ));

        return new HomePage(driver);
    }

    /**
     * Login fallido - espera que aparezca el modal de error.
     */
    public void loginFail(String email, String password) {
        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(20));

        WebElement emailField = wait.until(
                ExpectedConditions.elementToBeClickable(By.name("email"))
        );
        emailField.clear();
        emailField.sendKeys(email);

        WebElement passwordField = wait.until(
                ExpectedConditions.elementToBeClickable(By.name("password"))
        );
        passwordField.clear();
        passwordField.sendKeys(password);

        WebElement loginBtn = wait.until(
                ExpectedConditions.elementToBeClickable(By.cssSelector("button[type='submit']"))
        );
        loginBtn.click();

        wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector(".modal-content")));
    }

    public boolean isErrorModalDisplayed() {
        try {
            WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(5));
            wait.until(ExpectedConditions.visibilityOfElementLocated(By.cssSelector(".modal-content")));
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public String getErrorMessage() {
        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(5));
        WebElement modal = wait.until(
                ExpectedConditions.visibilityOfElementLocated(By.cssSelector(".modal-content"))
        );
        return modal.getText();
    }

    public RegisterPage clickRegisterLink() {
        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));
        WebElement link = wait.until(ExpectedConditions.elementToBeClickable(registerLink));
        link.click();
        return new RegisterPage(driver);
    }
}