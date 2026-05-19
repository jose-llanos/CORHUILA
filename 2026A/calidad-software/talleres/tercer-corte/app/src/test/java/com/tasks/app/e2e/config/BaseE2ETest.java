package com.tasks.app.e2e.config;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.extension.ExtendWith;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

/**
 * Base para todos los tests E2E. Subclases siguen este patrón:
 *
 * <pre>{@code
 * @ParameterizedTest(name = "{0}")
 * @EnumSource(BrowserType.class)
 * void mi_test(BrowserType browser) {
 *     setupDriver(browser);
 *     // ... usar driver, wait, baseUrl ...
 * }
 * }</pre>
 *
 * <p>El driver NO se crea en {@code @BeforeEach} porque ahí aún no
 * tenemos el {@link BrowserType} (JUnit inyecta parámetros a métodos
 * de test parametrizados, no a callbacks). Por eso usamos
 * {@link #setupDriver(BrowserType)} como primera línea del test.</p>
 */
@ExtendWith(ScreenshotOnFailure.class)
public abstract class BaseE2ETest {

    protected static final Duration DEFAULT_WAIT = Duration.ofSeconds(10);

    /**
     * Driver actual del test. ThreadLocal porque Surefire puede
     * paralelizar y porque la extensión de screenshot lo lee desde fuera.
     */
    private static final ThreadLocal<WebDriver> CURRENT_DRIVER = new ThreadLocal<>();

    protected WebDriver driver;
    protected WebDriverWait wait;
    protected String baseUrl;

    protected void setupDriver(BrowserType browser) {
        this.driver = WebDriverFactory.create(browser);
        this.wait = new WebDriverWait(driver, DEFAULT_WAIT);
        this.baseUrl = System.getProperty("e2e.baseUrl", "http://app:8080");
        CURRENT_DRIVER.set(driver);
    }

    @AfterEach
    void tearDown() {
        try {
            if (driver != null) driver.quit();
        } finally {
            CURRENT_DRIVER.remove();
            driver = null;
        }
    }

    /** Accesor para la extensión de screenshot. */
    static WebDriver currentDriver() {
        return CURRENT_DRIVER.get();
    }
}
