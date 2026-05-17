package com.autospark.migueljuliana.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.text.Normalizer;
import java.time.Duration;
import java.util.List;

public class ServicesPage extends BasePage {

    private static final String SERVICES_URL = "http://autospark_frontend:4200/services";

    private final By servicesGrid = By.cssSelector(".servicios-grid");
    private final By serviceCards = By.cssSelector(".servicios-grid .servicio");
    private final By serviceTitles = By.cssSelector(".servicios-grid .servicio h3");

    public ServicesPage(WebDriver driver) {
        super(driver);
    }

    public void navigateTo() {
        driver.get(SERVICES_URL);

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(30));

        wait.until(ExpectedConditions.presenceOfElementLocated(servicesGrid));

        wait.until(ExpectedConditions.numberOfElementsToBeMoreThan(
                serviceCards,
                0
        ));

        System.out.println("Página de servicios cargada: " + driver.getCurrentUrl());
        System.out.println("Cantidad de servicios visibles: " + getServiceCount());
    }

    public int getServiceCount() {
        try {
            WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(30));

            wait.until(ExpectedConditions.numberOfElementsToBeMoreThan(
                    serviceCards,
                    0
            ));

            List<WebElement> cards = driver.findElements(serviceCards);

            System.out.println("Servicios encontrados: " + cards.size());

            return cards.size();

        } catch (Exception e) {
            System.out.println("No se encontraron servicios: " + e.getMessage());
            System.out.println("URL actual: " + driver.getCurrentUrl());
            System.out.println("HTML actual:");
            System.out.println(driver.getPageSource());
            return 0;
        }
    }

    public boolean isServiceDisplayed(String serviceName) {
        try {
            WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(30));

            wait.until(ExpectedConditions.numberOfElementsToBeMoreThan(
                    serviceCards,
                    0
            ));

            List<WebElement> titles = driver.findElements(serviceTitles);

            String expected = normalize(serviceName);

            for (WebElement title : titles) {
                String actualText = title.getText().trim();
                String actual = normalize(actualText);

                System.out.println("Nombre de servicio encontrado: '" + actualText + "'");

                if (actual.contains(expected) || expected.contains(actual)) {
                    return true;
                }
            }

            return false;

        } catch (Exception e) {
            System.out.println("Error buscando servicio: " + e.getMessage());
            System.out.println("URL actual: " + driver.getCurrentUrl());
            System.out.println("HTML actual:");
            System.out.println(driver.getPageSource());
            return false;
        }
    }

    private String normalize(String text) {
        if (text == null) {
            return "";
        }

        return Normalizer.normalize(text, Normalizer.Form.NFD)
                .replaceAll("\\p{M}", "")
                .trim()
                .toLowerCase();
    }
}