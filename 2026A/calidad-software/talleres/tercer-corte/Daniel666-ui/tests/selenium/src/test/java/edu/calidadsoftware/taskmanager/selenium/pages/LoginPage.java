package edu.calidadsoftware.taskmanager.selenium.pages;

import edu.calidadsoftware.taskmanager.selenium.config.TestConfig;
import org.openqa.selenium.By;

/**
 * Page Object: Login.
 */
public class LoginPage extends BasePage {

    private final By usernameInput = By.id("username");
    private final By passwordInput = By.id("password");
    private final By errorAlert = By.cssSelector(".alert.alert-error");

    public LoginPage open() {
        driver.get(TestConfig.baseUrl() + "/login");
        waitVisible(usernameInput);
        return this;
    }

    public DashboardPage loginValid(String username, String password) {
        type(usernameInput, username);
        type(passwordInput, password);
        waitClickable(By.cssSelector("button[type='submit']")).click();
        return new DashboardPage().waitUntilLoaded();
    }

    public LoginPage loginInvalid(String username, String password) {
        type(usernameInput, username);
        type(passwordInput, password);
        waitClickable(By.cssSelector("button[type='submit']")).click();
        waitVisible(errorAlert);
        return this;
    }

    public String getErrorMessage() {
        return waitVisible(errorAlert).getText().trim();
    }
}

