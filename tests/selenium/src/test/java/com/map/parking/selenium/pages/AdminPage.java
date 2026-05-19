package com.map.parking.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;

public class AdminPage extends BasePage {

    private static final By ROLE_LABEL = By.xpath("//p[contains(@class,'text-muted') and contains(.,'Administrador')]");

    public AdminPage(WebDriver driver) {
        super(driver);
    }

    public boolean isAdminPanelVisible() {
        waitUrlContains("/admin");
        return waitVisible(ROLE_LABEL).isDisplayed();
    }
}
