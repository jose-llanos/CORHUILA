package com.tasks.app.e2e.pages;
 
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.support.ui.WebDriverWait;
 
public class ProfileModalPage extends BasePage {
 
    public ProfileModalPage(WebDriver driver, WebDriverWait wait, String baseUrl) {
        super(driver, wait, baseUrl);
        waitVisible(byTest("profile-info-list"));
    }
 
    public String getId()        { return text(byTest("profile-id")); }
    public String getUsername()  { return text(byTest("profile-username")); }
    public String getEmail()     { return text(byTest("profile-email")); }
    public String getCreatedAt() { return text(byTest("profile-created-at")); }
}
 