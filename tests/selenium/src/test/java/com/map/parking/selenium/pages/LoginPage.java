package com.map.parking.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;

public class LoginPage extends BasePage {

    private static final By EMAIL_INPUT = By.cssSelector("input[name='email']");
    private static final By PASSWORD_INPUT = By.cssSelector("input[name='password']");
    private static final By SUBMIT_BUTTON = By.xpath("//button[@type='submit' and contains(.,'Iniciar sesión')]");
    private static final By REGISTER_BUTTON = By.xpath("//button[contains(.,'Registrar')]");
    private static final By RECOVER_LINK = By.xpath("//a[contains(@href,'recuperarcontrasenia') or contains(.,'aquí')]");
    private static final By ERROR_ALERT = By.cssSelector(".alert.alert-danger");

    public LoginPage(WebDriver driver) {
        super(driver);
    }

    public LoginPage open() {
        open("/login");
        waitVisible(EMAIL_INPUT);
        return this;
    }

    public LoginPage enterEmail(String email) {
        type(EMAIL_INPUT, email);
        return this;
    }

    public LoginPage enterPassword(String password) {
        type(PASSWORD_INPUT, password);
        return this;
    }

    public LoginPage submit() {
        click(SUBMIT_BUTTON);
        return this;
    }

    public LoginPage login(String email, String password) {
        enterEmail(email);
        enterPassword(password);
        return submit();
    }

    public String getErrorMessage() {
        return waitVisible(ERROR_ALERT).getText();
    }

    public boolean isErrorDisplayed() {
        return !driver.findElements(ERROR_ALERT).isEmpty()
                && driver.findElement(ERROR_ALERT).isDisplayed();
    }

    public String waitForErrorMessage() {
        return waitVisible(ERROR_ALERT).getText();
    }

    public RegisterPage goToRegister() {
        click(REGISTER_BUTTON);
        return new RegisterPage(driver);
    }

    public RecoverPasswordPage goToRecoverPassword() {
        click(RECOVER_LINK);
        return new RecoverPasswordPage(driver);
    }
}
