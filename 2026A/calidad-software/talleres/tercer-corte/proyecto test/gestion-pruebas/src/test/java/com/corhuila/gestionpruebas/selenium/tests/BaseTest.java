package com.corhuila.gestionpruebas.selenium.tests;

import io.github.bonigarcia.wdm.WebDriverManager;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.firefox.FirefoxOptions;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import java.time.Duration;

/**
 * Clase base para todas las pruebas Selenium.
 * Soporta Chrome y Firefox mediante la propiedad de sistema -Dbrowser=firefox
 *
 * Ejecución:
 *   Chrome (defecto): mvn test -Dtest=VeterinariaTest
 *   Firefox:          mvn test -Dtest=VeterinariaTest -Dbrowser=firefox
 */
public class BaseTest {

    protected static WebDriver driver;
    protected static String baseUrl = "http://localhost:8081";

    private static final String USERNAME = "admin";
    private static final String PASSWORD = "admin123";

    @BeforeAll
    public static void setupClass() {
        String browser = System.getProperty("browser", "chrome").toLowerCase();

        if (browser.equals("firefox")) {
            setupFirefox();
        } else {
            setupChrome();
        }

        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));
        driver.manage().window().maximize();

        loginAutomatico();
    }

    // ─── Configuración Chrome ───────────────────────────
    private static void setupChrome() {
        WebDriverManager.chromedriver().driverVersion("147.0.7727.116").setup();
        ChromeOptions options = new ChromeOptions();
        options.addArguments("--no-sandbox");
        options.addArguments("--disable-dev-shm-usage");
        options.addArguments("--remote-allow-origins=*");
        options.addArguments("--window-size=1280,720");
        // Descomentar para CI/CD sin pantalla:
        // options.addArguments("--headless=new");
        System.out.println("🚀 Iniciando Chrome...");
        driver = new ChromeDriver(options);
    }

    // ─── Configuración Firefox ──────────────────────────
    private static void setupFirefox() {
        WebDriverManager.firefoxdriver().setup();
        FirefoxOptions options = new FirefoxOptions();
        options.addArguments("--width=1280");
        options.addArguments("--height=720");
        // Descomentar para CI/CD sin pantalla:
        // options.addArguments("--headless");
        System.out.println("🦊 Iniciando Firefox...");
        driver = new FirefoxDriver(options);
    }

    // ─── Login automático ───────────────────────────────
    private static void loginAutomatico() {
        try {
            System.out.println("🔐 Realizando login automático...");
            driver.get(baseUrl + "/login");

            WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(15));

            // Esperar a que cargue el formulario de login
            WebElement usernameField = wait.until(
                    ExpectedConditions.visibilityOfElementLocated(By.name("username"))
            );

            // Ingresar credenciales
            usernameField.clear();
            usernameField.sendKeys(USERNAME);

            WebElement passwordField = driver.findElement(By.name("password"));
            passwordField.clear();
            passwordField.sendKeys(PASSWORD);

            // Hacer clic en el botón de login
            driver.findElement(By.cssSelector("button[type='submit']")).click();

            // Esperar a que se complete el login (redirige fuera de /login)
            wait.until(ExpectedConditions.not(
                    ExpectedConditions.urlContains("/login")
            ));

            System.out.println("✅ Login exitoso. URL actual: " + driver.getCurrentUrl());

        } catch (Exception e) {
            System.err.println("❌ Error en login automático: " + e.getMessage());
            throw new RuntimeException("No se pudo realizar login en " + baseUrl + "/login", e);
        }
    }

    @AfterAll
    public static void tearDown() {
        if (driver != null) {
            System.out.println("🔚 Cerrando navegador...");
            driver.quit();
        }
    }
}