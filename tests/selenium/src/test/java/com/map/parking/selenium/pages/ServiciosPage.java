package com.map.parking.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;

public class ServiciosPage extends BasePage {

    private static final By PAGE_TITLE = By.xpath("//h1[contains(.,'Nuestros Servicios')]");

    public ServiciosPage(WebDriver driver) {
        super(driver);
    }

    public String getPageTitle() {
        return waitVisible(PAGE_TITLE).getText();
    }
}
