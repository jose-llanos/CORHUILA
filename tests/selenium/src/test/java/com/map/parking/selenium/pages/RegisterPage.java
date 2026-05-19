package com.map.parking.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;

public class RegisterPage extends BasePage {

    private static final By PAGE_TITLE = By.xpath("//h2[contains(.,'REGISTRO')]");

    public RegisterPage(WebDriver driver) {
        super(driver);
    }

    public String getPageTitle() {
        return waitVisible(PAGE_TITLE).getText();
    }
}
