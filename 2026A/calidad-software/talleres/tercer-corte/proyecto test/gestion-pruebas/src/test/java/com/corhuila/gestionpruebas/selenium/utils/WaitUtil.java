package com.corhuila.gestionpruebas.selenium.utils;

import org.openqa.selenium.*;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import java.time.Duration;
import java.util.function.Function;

public class WaitUtil {

    private static final int DEFAULT_TIMEOUT = 10;
    private static final int POLLING_INTERVAL = 1;

    public static WebDriverWait getWait(WebDriver driver) {
        return new WebDriverWait(driver, Duration.ofSeconds(DEFAULT_TIMEOUT),
                Duration.ofSeconds(POLLING_INTERVAL));
    }

    public static WebDriverWait getWait(WebDriver driver, int timeoutSegundos) {
        return new WebDriverWait(driver, Duration.ofSeconds(timeoutSegundos),
                Duration.ofSeconds(POLLING_INTERVAL));
    }

    public static void esperarElementoVisible(WebDriver driver, By locator) {
        getWait(driver).until(ExpectedConditions.visibilityOfElementLocated(locator));
    }

    public static void esperarElementoClickeable(WebDriver driver, By locator) {
        getWait(driver).until(ExpectedConditions.elementToBeClickable(locator));
    }

    public static void esperarTextoPresente(WebDriver driver, By locator, String texto) {
        getWait(driver).until(ExpectedConditions.textToBePresentInElementLocated(locator, texto));
    }

    public static void esperarAlertPresente(WebDriver driver) {
        getWait(driver).until(ExpectedConditions.alertIsPresent());
    }

    public static boolean esperarAtributo(WebDriver driver, WebElement element,
                                          String atributo, String valor) {
        return getWait(driver).until(driver1 -> {
            String attrValue = element.getAttribute(atributo);
            return valor.equals(attrValue);
        });
    }

    public static void esperarPaginaCargada(WebDriver driver) {
        getWait(driver).until(driver1 ->
                ((JavascriptExecutor) driver1).executeScript("return document.readyState").equals("complete")
        );
    }
}