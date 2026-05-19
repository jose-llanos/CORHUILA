package com.tasks.app.e2e.config;

import org.openqa.selenium.MutableCapabilities;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.firefox.FirefoxOptions;
import org.openqa.selenium.remote.RemoteWebDriver;

import java.net.MalformedURLException;
import java.net.URL;
import java.time.Duration;

/**
 * Construye un RemoteWebDriver apuntando al Selenium Grid.
 *
 * <p>La URL del Grid se lee de la system property {@code e2e.gridUrl} y
 * por defecto es {@code http://selenium-hub:4444} (hostname interno de la red
 * qa-net en Docker). Para correr local pasa
 * {@code -De2e.gridUrl=http://localhost:4444}.</p>
 *
 * <p>Modo headless por defecto: Jenkins no tiene display.</p>
 */
public final class WebDriverFactory {

    private static final String DEFAULT_GRID_URL = "http://selenium-hub:4444";
    private static final Duration IMPLICIT_WAIT = Duration.ofSeconds(2);

    private WebDriverFactory() {}

    public static WebDriver create(BrowserType browser) {
        String gridUrl = System.getProperty("e2e.gridUrl", DEFAULT_GRID_URL);
        MutableCapabilities options = buildOptions(browser);

        try {
            RemoteWebDriver driver = new RemoteWebDriver(new URL(gridUrl), options);
            driver.manage().timeouts().implicitlyWait(IMPLICIT_WAIT);
            driver.manage().window().setSize(new org.openqa.selenium.Dimension(1366, 900));
            return driver;
        } catch (MalformedURLException e) {
            throw new IllegalStateException("URL del Selenium Grid inválida: " + gridUrl, e);
        }
    }

    private static MutableCapabilities buildOptions(BrowserType browser) {
        return switch (browser) {
            case CHROME -> {
                ChromeOptions opts = new ChromeOptions();
                opts.addArguments("--headless=new");
                opts.addArguments("--no-sandbox");
                opts.addArguments("--disable-dev-shm-usage");
                opts.addArguments("--disable-gpu");
                opts.addArguments("--window-size=1366,900");
                yield opts;
            }
            case FIREFOX -> {
                FirefoxOptions opts = new FirefoxOptions();
                opts.addArguments("-headless");
                yield opts;
            }
        };
    }
}
