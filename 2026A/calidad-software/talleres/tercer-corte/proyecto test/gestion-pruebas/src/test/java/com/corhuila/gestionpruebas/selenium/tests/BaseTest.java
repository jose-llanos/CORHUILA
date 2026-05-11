package com.corhuila.gestionpruebas.selenium.tests;
import io.github.bonigarcia.wdm.WebDriverManager;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.openqa.selenium.support.ui.ExpectedConditions;
import java.time.Duration;

public class BaseTest {
    protected static WebDriver driver;
    protected static String baseUrl = "http://localhost:8081";

    @BeforeAll
    public static void setupClass() {
        WebDriverManager.chromedriver().driverVersion("147.0.7727.116").setup();

        ChromeOptions options = new ChromeOptions();
        options.addArguments("--no-sandbox");
        options.addArguments("--disable-dev-shm-usage");
        options.addArguments("--remote-allow-origins=*");
        options.addArguments("--window-size=1280,720");
        // ✅ Sin setBinary — ChromeDriver detecta Chromium automáticamente

        System.out.println("🚀 Iniciando Chrome...");
        driver = new ChromeDriver(options);
        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));
        driver.manage().window().maximize();

        // Login
        driver.get("http://localhost:8081/login");
        System.out.println("⏳ Ingresa usuario y contraseña en el navegador...");
        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(60));
        wait.until(ExpectedConditions.not(
                ExpectedConditions.urlContains("/login")
        ));
        System.out.println("✅ Login exitoso — corriendo los tests...");
    }

    @AfterAll
    public static void tearDown() {
        if (driver != null) {
            System.out.println("🔚 Cerrando navegador...");
            driver.quit();
        }
    }
}