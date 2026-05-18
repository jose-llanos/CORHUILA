package com.medicita.selenium.base;

import io.github.bonigarcia.wdm.WebDriverManager;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;

import java.time.Duration;

/*
 * Clase base para todos los tests de Selenium.
 * Se encarga de levantar y cerrar el ChromeDriver en cada test.
 * Usa --headless para que corra sin abrir ventana visible (ideal para CI/CD).
 * WebDriverManager descarga el ChromeDriver correcto automáticamente, sin
 * necesidad de instalar nada a mano.
 */
public abstract class BaseTest {

    protected WebDriver driver;

    // URL base leída de la propiedad del sistema; por defecto apunta a localhost (Docker)
    protected static final String BASE_URL =
            System.getProperty("app.url", "http://localhost");

    @BeforeEach
    void setUpDriver() {
        WebDriverManager.chromedriver().setup();

        ChromeOptions options = new ChromeOptions();
        options.addArguments("--headless=new");           // sin ventana visible
        options.addArguments("--no-sandbox");              // necesario en Linux/CI
        options.addArguments("--disable-dev-shm-usage");  // evita crashes en Docker
        options.addArguments("--window-size=1280,800");
        options.addArguments("--disable-gpu");

        driver = new ChromeDriver(options);
        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(8));
        driver.manage().timeouts().pageLoadTimeout(Duration.ofSeconds(30));
    }

    @AfterEach
    void tearDownDriver() {
        if (driver != null) {
            driver.quit();
        }
    }
}
