package com.panaderia.panaderia.selenium.tests;

import com.panaderia.panaderia.selenium.pages.HomePage;

import io.github.bonigarcia.wdm.WebDriverManager;

import org.junit.jupiter.api.*;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.*;

class PanaderiaSeleniumTest {

    private WebDriver driver;
    private HomePage homePage;
    private WebDriverWait wait;

    @BeforeEach
    void setUp() {

        WebDriverManager.chromedriver().setup();

        driver = new ChromeDriver();

        driver.manage().window().maximize();

        wait = new WebDriverWait(driver, Duration.ofSeconds(10));

        homePage = new HomePage(driver);
    }

    @AfterEach
    void tearDown() {

        if(driver != null){
            driver.quit();
        }

    }

    // TEST 1
    @Test
    void debeCargarPaginaPrincipal() {

        homePage.abrir();

        assertTrue(homePage.paginaCargada());

    }

    // TEST 2
    @Test
    void debeMostrarProductos() {

        homePage.abrir();

        assertTrue(homePage.existenProductos());

    }

    // TEST 3
    @Test
    void debeAgregarProductoAlCarrito() {

        homePage.abrir();

        homePage.agregarPrimerProducto();

        assertTrue(driver.getPageSource().contains("Pan"));

    }

    // TEST 4
    @Test
    void debeAbrirCarrito() {

        homePage.abrir();

        homePage.irACarrito();

        assertTrue(driver.getCurrentUrl().contains("carrito"));

    }

    // TEST 5
    @Test
    void debeAbrirInventario() {

        homePage.abrir();

        homePage.irAInventario();

        assertTrue(driver.getCurrentUrl().contains("inventario"));

    }

}