package com.tasks.app.e2e.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

/**
 * Base de todos los Page Objects. Encapsula:
 * <ul>
 *   <li>Construcción de selectores por {@code data-test}.</li>
 *   <li>Esperas explícitas (nunca {@code Thread.sleep}).</li>
 *   <li>Acciones atómicas: type, click, getText.</li>
 * </ul>
 */
public abstract class BasePage {

    protected final WebDriver driver;
    protected final WebDriverWait wait;
    protected final String baseUrl;

    protected BasePage(WebDriver driver, WebDriverWait wait, String baseUrl) {
        this.driver = driver;
        this.wait = wait;
        this.baseUrl = baseUrl;
    }

    /** Selector CSS por atributo data-test. */
    protected static By byTest(String value) {
        return By.cssSelector("[data-test='" + value + "']");
    }

    protected WebElement waitVisible(By by) {
        return wait.until(ExpectedConditions.visibilityOfElementLocated(by));
    }

    protected WebElement waitClickable(By by) {
        return wait.until(ExpectedConditions.elementToBeClickable(by));
    }

    protected void waitInvisible(By by) {
        wait.until(ExpectedConditions.invisibilityOfElementLocated(by));
    }

    protected void type(By by, String text) {
        WebElement el = waitVisible(by);
        el.clear();
        el.sendKeys(text);
    }

    protected void click(By by) {
        waitClickable(by).click();
    }

    protected String text(By by) {
        return waitVisible(by).getText();
    }

    protected boolean isPresent(By by) {
        return !driver.findElements(by).isEmpty();
    }
}
