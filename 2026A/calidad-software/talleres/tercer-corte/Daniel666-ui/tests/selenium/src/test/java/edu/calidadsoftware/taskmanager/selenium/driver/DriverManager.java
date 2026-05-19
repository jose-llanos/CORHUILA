package edu.calidadsoftware.taskmanager.selenium.driver;

import org.openqa.selenium.WebDriver;

/**
 * Maneja el WebDriver por hilo (ThreadLocal).
 *
 * Esto permite paralelismo en el futuro y evita colisiones entre pruebas.
 */
public final class DriverManager {

    private static final ThreadLocal<WebDriver> DRIVER = new ThreadLocal<>();

    private DriverManager() {
    }

    public static void set(WebDriver driver) {
        DRIVER.set(driver);
    }

    public static WebDriver get() {
        return DRIVER.get();
    }

    public static void clear() {
        DRIVER.remove();
    }
}
