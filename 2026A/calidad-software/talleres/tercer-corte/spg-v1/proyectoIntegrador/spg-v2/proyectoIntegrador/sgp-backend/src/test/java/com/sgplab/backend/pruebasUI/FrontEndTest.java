package com.sgplab.backend.pruebasUI;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.remote.RemoteWebDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.net.MalformedURLException;
import java.net.URL;
import java.time.Duration;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class FrontEndTest {

    private WebDriver driver;

    @BeforeEach
    public void setUp() throws MalformedURLException {
        ChromeOptions options = new ChromeOptions();
        driver = new RemoteWebDriver(new URL("http://localhost:4444/wd/hub"), options);
    }

    @Test
    public void probarCargaDelFront() {
        // 1. Entrar a la página (Usa la IP que te funcionó en el paso anterior)
        driver.get("http://192.168.101.10:5173/");

        // 2. Esperar hasta 5 segundos a que el título contenga la palabra "SGP"
        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(5));
        wait.until(ExpectedConditions.titleContains("SGP"));

        // 3. Ahora sí, leer y verificar el título
        String pageTitle = driver.getTitle();
        System.out.println("El título de la página es: " + pageTitle);

        assertTrue(pageTitle != null && pageTitle.contains("SGP Lab"),
                "El título no coincide con lo esperado.");
    }

    @AfterEach
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}