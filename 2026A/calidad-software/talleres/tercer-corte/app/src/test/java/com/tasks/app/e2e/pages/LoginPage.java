package com.tasks.app.e2e.pages;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.support.ui.WebDriverWait;

public class LoginPage extends BasePage {

    public LoginPage(WebDriver driver, WebDriverWait wait, String baseUrl) {
        super(driver, wait, baseUrl);
    }

    public LoginPage open() {
        driver.get(baseUrl + "/index.html");
        waitVisible(byTest("login-username"));
        return this;
    }

    public DashboardPage login(String username, String password) {
        type(byTest("login-username"), username);
        type(byTest("login-password"), password);
        click(byTest("btn-login-submit"));
        // Tras login el front redirige a /dashboard.html: esperamos un
        // elemento exclusivo del dashboard antes de devolver el POM.
        waitVisible(byTest("username-display"));
        return new DashboardPage(driver, wait, baseUrl);
    }

    public RegisterPage goToRegister() {
        click(byTest("link-to-register"));
        waitVisible(byTest("reg-username"));
        return new RegisterPage(driver, wait, baseUrl);
    }

    public String getErrorMessage() {
        return text(byTest("auth-error-message"));
    }
}
