package com.map.parking.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;

public class RecoverPasswordPage extends BasePage {

    private static final By PAGE_TITLE = By.xpath("//h2[contains(.,'Recuperar Contraseña')]");
    private static final By EMAIL_INPUT = By.cssSelector("input#email");

    public RecoverPasswordPage(WebDriver driver) {
        super(driver);
    }

    public String getPageTitle() {
        return waitVisible(PAGE_TITLE).getText();
    }

    public boolean isEmailFieldVisible() {
        return waitVisible(EMAIL_INPUT).isDisplayed();
    }
}
