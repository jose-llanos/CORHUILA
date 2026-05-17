package com.autospark.migueljuliana.selenium.pages;

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

    @FindBy(css = ".modal-content")
    private WebElement successModal;

    @FindBy(css = ".modal button")
    private WebElement modalCloseButton;

    public RegisterPage(WebDriver driver) {
        super(driver);
    }

    public void navigateTo() {
        driver.get("http://autospark_frontend:4200/register");
    }

    public void register(String fullName, String email, String password,
                         String identityCard, String phone, String licensePlate) {
        wait.until(ExpectedConditions.elementToBeClickable(fullNameInput));

        fullNameInput.sendKeys(fullName);
        emailInput.sendKeys(email);
        passwordInput.sendKeys(password);
        identityCardInput.sendKeys(identityCard);
        phoneInput.sendKeys(phone);
        licensePlateInput.sendKeys(licensePlate);

        submitButton.click();
    }

    public void registerWithRole(String fullName, String email, String password,
                                 String identityCard, String phone, String licensePlate,
                                 String role) {
        wait.until(ExpectedConditions.elementToBeClickable(fullNameInput));

        fullNameInput.sendKeys(fullName);
        emailInput.sendKeys(email);
        passwordInput.sendKeys(password);
        identityCardInput.sendKeys(identityCard);
        phoneInput.sendKeys(phone);
        licensePlateInput.sendKeys(licensePlate);

        Select roleSelector = new Select(roleSelect);
        roleSelector.selectByVisibleText(role);

        submitButton.click();
    }

    public boolean isSuccessModalDisplayed() {
        try {
            wait.until(ExpectedConditions.visibilityOf(successModal));
            return successModal.isDisplayed();
        } catch (Exception e) {
            return false;
        }
    }

    public String getSuccessMessage() {
        return successModal.getText();
    }

    public void closeModal() {
        wait.until(ExpectedConditions.elementToBeClickable(modalCloseButton)).click();
    }
}