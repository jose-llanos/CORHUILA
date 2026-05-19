package edu.calidadsoftware.taskmanager.selenium.pages;

import edu.calidadsoftware.taskmanager.selenium.config.TestConfig;
import edu.calidadsoftware.taskmanager.selenium.driver.DriverManager;
import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

/**
 * Page Object base con utilidades de espera explícita.
 *
 * Nota: se evita Thread.sleep y se utilizan esperas con WebDriverWait.
 */
public abstract class BasePage {

    protected final WebDriver driver;
    protected final WebDriverWait wait;

    protected BasePage() {
        this.driver = DriverManager.get();
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(TestConfig.timeoutSeconds()));
    }

    protected WebElement waitVisible(By locator) {
        return wait.until(ExpectedConditions.visibilityOfElementLocated(locator));
    }

    protected WebElement waitClickable(By locator) {
        return wait.until(ExpectedConditions.elementToBeClickable(locator));
    }

    protected void type(By locator, String value) {
        WebElement el = waitVisible(locator);
        el.clear();
        if (value != null) {
            el.sendKeys(value);
        }
    }

    protected void waitUrlContains(String fragment) {
        wait.until(ExpectedConditions.urlContains(fragment));
    }

    protected void waitDocumentReady() {
        wait.until(d -> {
            if (!(d instanceof JavascriptExecutor)) {
                return true;
            }
            Object state = ((JavascriptExecutor) d).executeScript("return document.readyState");
            return "complete".equals(String.valueOf(state));
        });
    }
}
