package com.corhuila.gestionpruebas.selenium.tests;

import com.corhuila.gestionpruebas.selenium.pages.CitasPage;
import com.corhuila.gestionpruebas.selenium.pages.DuenosPage;
import com.corhuila.gestionpruebas.selenium.pages.MascotasPage;
import com.corhuila.gestionpruebas.selenium.utils.DataGenerator;

import io.github.bonigarcia.wdm.WebDriverManager;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.MethodOrderer.OrderAnnotation;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.firefox.FirefoxOptions;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Suite de pruebas funcionales en Firefox.
 * Cubre los mismos flujos críticos que VeterinariaTest pero con Firefox.
 * Requerido por: Especificación 5.3 — "Pruebas en navegadores Chrome y Firefox"
 */
@TestMethodOrder(OrderAnnotation.class)
public class VeterinariaTestFirefox {

    private static WebDriver driver;
    private static final String BASE_URL = "http://localhost:8081";

    @BeforeAll
    static void setup() {
        WebDriverManager.firefoxdriver().setup();
        FirefoxOptions options = new FirefoxOptions();
        options.addArguments("--width=1280", "--height=720");
        // options.addArguments("--headless"); // para CI sin pantalla
        driver = new FirefoxDriver(options);
        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(10));
        driver.manage().window().maximize();
        loginFirefox();
    }

    private static void loginFirefox() {
        driver.get(BASE_URL + "/login");
        WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(15));
        WebElement user = wait.until(ExpectedConditions.visibilityOfElementLocated(By.name("username")));
        user.sendKeys("admin");
        driver.findElement(By.name("password")).sendKeys("admin123");
        driver.findElement(By.cssSelector("button[type='submit']")).click();
        wait.until(ExpectedConditions.not(ExpectedConditions.urlContains("/login")));
        System.out.println("✅ Login Firefox exitoso");
    }

    @AfterAll
    static void tearDown() {
        if (driver != null) driver.quit();
    }

    // ───────────────────────────────────────────────────────────────────

    @Test
    @Order(1)
    @DisplayName("FF-01: Crear dueño en Firefox")
    void ff01_crearDueno() {
        driver.get(BASE_URL + "/duenios");
        DuenosPage page = new DuenosPage(driver);
        String nombre = DataGenerator.generarNombre();
        page.irANuevoDueno();
        page.crearDueno(nombre, DataGenerator.generarTelefono(), DataGenerator.generarEmail());
        assertTrue(page.existeDuenoEnTabla(nombre),
                "El dueño creado debe aparecer en la tabla");
    }

    @Test
    @Order(2)
    @DisplayName("FF-02: Crear mascota en Firefox")
    void ff02_crearMascota() {
        // Asegurar que hay al menos un dueño
        driver.get(BASE_URL + "/duenios");
        DuenosPage duenosPage = new DuenosPage(driver);
        duenosPage.irANuevoDueno();
        duenosPage.crearDueno(
                DataGenerator.generarNombre(),
                DataGenerator.generarTelefono(),
                DataGenerator.generarEmail()
        );

        driver.get(BASE_URL + "/mascotas");
        MascotasPage page = new MascotasPage(driver);
        String nombre = DataGenerator.generarNombreMascota();
        page.irANuevaMascota();
        page.crearMascota(nombre, "Gato", "Siamés", "2");
        assertTrue(driver.getPageSource().contains(nombre),
                "La mascota debe aparecer en la lista");
    }

    @Test
    @Order(3)
    @DisplayName("FF-03: Validar campos requeridos en Firefox")
    void ff03_validarCamposRequeridos() {
        driver.get(BASE_URL + "/duenios/nuevo");
        driver.findElement(By.cssSelector("button[type='submit']")).click();
        assertTrue(driver.getPageSource().contains("required"),
                "El formulario debe tener validación HTML5");
    }

    @Test
    @Order(4)
    @DisplayName("FF-04: Acceso al módulo de citas en Firefox")
    void ff04_accesoModuloCitas() {
        driver.get(BASE_URL + "/citas");
        CitasPage page = new CitasPage(driver);
        assertTrue(driver.getPageSource().contains("Citas"));
        page.irANuevaCita();
        assertTrue(driver.getCurrentUrl().contains("/citas"));
    }

    @Test
    @Order(5)
    @DisplayName("FF-05: Navegación completa entre módulos en Firefox")
    void ff05_navegacionCompleta() {
        driver.get(BASE_URL + "/duenios");
        assertTrue(driver.getPageSource().contains("Dueños") ||
                driver.getPageSource().contains("Due"), "Módulo Dueños accesible");

        driver.get(BASE_URL + "/mascotas");
        assertTrue(driver.getPageSource().contains("Mascotas"), "Módulo Mascotas accesible");

        driver.get(BASE_URL + "/citas");
        assertTrue(driver.getPageSource().contains("Citas"), "Módulo Citas accesible");

        driver.get(BASE_URL + "/tratamientos");
        assertTrue(driver.getPageSource().contains("Tratamientos"), "Módulo Tratamientos accesible");
    }
}