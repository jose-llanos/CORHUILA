package com.corhuila.gestionpruebas.selenium.pages;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.support.PageFactory;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.By;
import java.time.Duration;

public abstract class BasePage {
    protected WebDriver driver;
    protected WebDriverWait wait;

    public BasePage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
        PageFactory.initElements(driver, this);
    }

    // Métodos comunes
    public void esperarElementoVisible(By locator) {
        wait.until(ExpectedConditions.visibilityOfElementLocated(locator));
    }

    public void esperarElementoClickeable(By locator) {
        wait.until(ExpectedConditions.elementToBeClickable(locator));
    }

    public String getTituloPagina() {
        return driver.getTitle();
    }

    public void tomarPantalla(String nombre) {
        // Método opcional para screenshots
    }
}