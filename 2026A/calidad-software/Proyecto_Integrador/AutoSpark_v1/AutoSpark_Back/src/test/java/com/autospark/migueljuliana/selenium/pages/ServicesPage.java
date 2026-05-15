package com.autospark.migueljuliana.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.FindBy;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import java.time.Duration;
import java.util.List;

public class ServicesPage extends BasePage {

    @FindBy(css = ".servicio")
    private List<WebElement> serviceCards;

    public ServicesPage(WebDriver driver) {
        super(driver);
    }

    public void navigateTo() {
        driver.get("http://localhost:4200/services");
        // Esperar a que Angular cargue completamente
        try { Thread.sleep(3000); } catch (InterruptedException e) {}
    }

    public int getServiceCount() {
        try {
            // Esperar explícitamente a que aparezcan los servicios
            WebDriverWait longWait = new WebDriverWait(driver, Duration.ofSeconds(15));
            longWait.until(ExpectedConditions.presenceOfAllElementsLocatedBy(
                    By.cssSelector(".servicio")));

            System.out.println("Servicios encontrados: " + serviceCards.size());
            return serviceCards.size();
        } catch (Exception e) {
            System.out.println("No se encontraron servicios: " + e.getMessage());
            return 0;
        }
    }

    public boolean isServiceDisplayed(String serviceName) {
        try {
            WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));
            List<WebElement> serviceNames = wait.until(
                    ExpectedConditions.presenceOfAllElementsLocatedBy(
                            By.xpath("//div[contains(@class, 'servicio')]//h3[1]")));

            for (WebElement name : serviceNames) {
                System.out.println("Nombre de servicio encontrado: '" + name.getText() + "'");
                if (name.getText().trim().equalsIgnoreCase(serviceName)) {
                    return true;
                }
            }
            return false;
        } catch (Exception e) {
            System.out.println("Error buscando servicio: " + e.getMessage());
            return false;
        }
    }
}