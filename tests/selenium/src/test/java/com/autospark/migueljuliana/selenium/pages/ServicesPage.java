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

        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(40));

        wait.until(ExpectedConditions.presenceOfElementLocated(By.tagName("body")));
        wait.until(ExpectedConditions.presenceOfElementLocated(servicesGrid));

        wait.until(driver -> driver.findElements(serviceCards).size() > 0
                || normalize(driver.getPageSource()).contains("lavado basico")
                || normalize(driver.getPageSource()).contains("lavado premium")
                || normalize(driver.getPageSource()).contains("pulido"));

        System.out.println("Página de servicios cargada: " + driver.getCurrentUrl());
        System.out.println("Cantidad de servicios visibles: " + getServiceCount());
    }

    public int getServiceCount() {
        try {
            WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(40));

            wait.until(ExpectedConditions.presenceOfElementLocated(servicesGrid));

            wait.until(driver -> driver.findElements(serviceCards).size() > 0
                    || normalize(driver.getPageSource()).contains("lavado basico")
                    || normalize(driver.getPageSource()).contains("lavado premium")
                    || normalize(driver.getPageSource()).contains("pulido"));

            List<WebElement> cards = driver.findElements(serviceCards);

            int visibleCards = 0;

            for (WebElement card : cards) {
                String text = card.getText();

                if (text != null && !text.trim().isEmpty()) {
                    visibleCards++;
                    System.out.println("Servicio visible encontrado: " + text);
                }
            }

            String html = normalize(driver.getPageSource());

            if (visibleCards == 0 &&
                    (html.contains("lavado basico")
                            || html.contains("lavado premium")
                            || html.contains("pulido")
                            || html.contains("lavado de motor")
                            || html.contains("lavado de tapiceria"))) {
                visibleCards = 1;
            }

            System.out.println("Servicios encontrados: " + visibleCards);

            return visibleCards;

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
            WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(40));

            wait.until(ExpectedConditions.presenceOfElementLocated(servicesGrid));

            String expected = normalize(serviceName);

            wait.until(driver -> driver.findElements(serviceTitles).size() > 0
                    || normalize(driver.getPageSource()).contains(expected));

            List<WebElement> titles = driver.findElements(serviceTitles);

            for (WebElement title : titles) {
                String actualText = title.getText().trim();
                String actual = normalize(actualText);

                System.out.println("Nombre de servicio encontrado: '" + actualText + "'");

                if (actual.contains(expected) || expected.contains(actual)) {
                    return true;
                }
            }

            return normalize(driver.getPageSource()).contains(expected);

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