package com.autospark.migueljuliana.selenium.pages;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.FindBy;
import org.openqa.selenium.support.ui.ExpectedConditions;

public class LoginPage extends BasePage {

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
        driver.get("http://localhost:4200/login");
    }

    public HomePage login(String email, String password) {
        wait.until(ExpectedConditions.elementToBeClickable(emailInput));

        emailInput.clear();
        emailInput.sendKeys(email);

        passwordInput.clear();
        passwordInput.sendKeys(password);

        loginButton.click();

        return new HomePage(driver);
    }

    public boolean isErrorModalDisplayed() {
        try {
            wait.until(ExpectedConditions.visibilityOf(errorModal));
            return errorModal.isDisplayed();
        } catch (Exception e) {
            return false;
        }
    }

    public String getErrorMessage() {
        return errorModal.getText();
    }

    public RegisterPage clickRegisterLink() {
        registerLink.click();
        return new RegisterPage(driver);
    }
}