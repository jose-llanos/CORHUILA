package com.map.parking.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;

public class HomePage extends BasePage {

    private static final By WELCOME_TITLE = By.cssSelector("h1.fw-bold.text-primary");
    private static final By LOGIN_BUTTON = By.xpath("//button[contains(normalize-space(),'Iniciar sesión')]");
    private static final By SERVICIOS_BUTTON = By.xpath("//button[contains(normalize-space(),'Ver servicios')]");

    public HomePage(WebDriver driver) {
        super(driver);
    }

    public HomePage open() {
        open("/home");
        return this;
    }

    public String getWelcomeText() {
        return waitVisible(WELCOME_TITLE).getText();
    }

    public LoginPage goToLogin() {
        click(LOGIN_BUTTON);
        return new LoginPage(driver);
    }

    public ServiciosPage goToServicios() {
        click(SERVICIOS_BUTTON);
        return new ServiciosPage(driver);
    }
}
