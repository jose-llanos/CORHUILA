package com.panaderia.panaderia.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import java.time.Duration;

public class HomePage {

    private WebDriver driver;
    private WebDriverWait wait;

    public HomePage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    public void abrir() {
        driver.get("http://localhost:4200");
    }

    public boolean paginaCargada() {
        return driver.getPageSource().contains("PANADERIA DULCE PAN");
    }

    public boolean existenProductos() {
        return driver.getPageSource().contains("Pan Coco");
    }

    public void agregarPrimerProducto() {
        wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("(//button[contains(.,'Agregar al carrito')])[1]")
        )).click();
    }

    public void irACarrito() {
        wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//a[contains(.,'CARRITO') or contains(.,'Carrito')]")
        )).click();
    }

    public void irAInventario() {
        wait.until(ExpectedConditions.elementToBeClickable(
                By.xpath("//a[contains(.,'INVENTARIO') or contains(.,'Inventario')]")
        )).click();
    }
}