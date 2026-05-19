package com.autospark.migueljuliana.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.FindBy;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.Select;

public class RegisterPage extends BasePage {

    @FindBy(name = "fullName")
    private WebElement fullNameInput;

    @FindBy(name = "email")
    private WebElement emailInput;

    @FindBy(name = "password")
    private WebElement passwordInput;

    @FindBy(name = "identityCard")
    private WebElement identityCardInput;

    @FindBy(name = "phone")
    private WebElement phoneInput;

    @FindBy(name = "licensePlate")
    private WebElement licensePlateInput;

    @FindBy(name = "role")
    private WebElement roleSelect;

    @FindBy(css = "button[type='submit']")
    private WebElement submitButton;

    private final By successModal = By.cssSelector(".modal-content");

    private final By modalCloseButton = By.cssSelector(".modal button");

    public RegisterPage(WebDriver driver) {
        super(driver);
    }

    public void navigateTo() {
        driver.get("http://autospark_frontend/register");
    }

    public void register(String fullName, String email, String password,
                         String identityCard, String phone, String licensePlate) {
        wait.until(ExpectedConditions.elementToBeClickable(fullNameInput));

        fullNameInput.clear();
        fullNameInput.sendKeys(fullName);

        emailInput.clear();
        emailInput.sendKeys(email);

        passwordInput.clear();
        passwordInput.sendKeys(password);

        identityCardInput.clear();
        identityCardInput.sendKeys(identityCard);

        phoneInput.clear();
        phoneInput.sendKeys(phone);

        licensePlateInput.clear();
        licensePlateInput.sendKeys(licensePlate);

        wait.until(ExpectedConditions.elementToBeClickable(submitButton)).click();
    }

    public void registerWithRole(String fullName, String email, String password,
                                 String identityCard, String phone, String licensePlate,
                                 String role) {
        wait.until(ExpectedConditions.elementToBeClickable(fullNameInput));

        fullNameInput.clear();
        fullNameInput.sendKeys(fullName);

        emailInput.clear();
        emailInput.sendKeys(email);

        passwordInput.clear();
        passwordInput.sendKeys(password);

        identityCardInput.clear();
        identityCardInput.sendKeys(identityCard);

        phoneInput.clear();
        phoneInput.sendKeys(phone);

        licensePlateInput.clear();
        licensePlateInput.sendKeys(licensePlate);

        Select roleSelector = new Select(roleSelect);
        roleSelector.selectByVisibleText(role);

        wait.until(ExpectedConditions.elementToBeClickable(submitButton)).click();
    }

    public boolean isSuccessModalDisplayed() {
        try {
            return wait.until(
                    ExpectedConditions.visibilityOfElementLocated(successModal)
            ).isDisplayed();
        } catch (Exception e) {
            return false;
        }
    }

    public String getSuccessMessage() {
        return wait.until(
                ExpectedConditions.visibilityOfElementLocated(successModal)
        ).getText();
    }

    public void closeModal() {
        wait.until(
                ExpectedConditions.elementToBeClickable(modalCloseButton)
        ).click();
    }
}