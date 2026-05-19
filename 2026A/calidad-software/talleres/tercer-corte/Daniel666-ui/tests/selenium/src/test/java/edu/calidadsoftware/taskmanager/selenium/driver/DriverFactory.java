package edu.calidadsoftware.taskmanager.selenium.driver;

import edu.calidadsoftware.taskmanager.selenium.config.TestConfig;
import io.github.bonigarcia.wdm.WebDriverManager;
import org.openqa.selenium.MutableCapabilities;
import org.openqa.selenium.Platform;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.firefox.FirefoxOptions;
import org.openqa.selenium.remote.RemoteWebDriver;

import java.net.URL;
import java.time.Duration;

/**
 * Factory de WebDriver.
 *
 * Soporta:
 * - Ejecución local con WebDriverManager (Chrome/Firefox)
 * - Ejecución remota con Selenium Grid (RemoteWebDriver)
 */
public final class DriverFactory {

    private DriverFactory() {
    }

    public static WebDriver createDriver() {
        String remoteUrl = TestConfig.remoteUrl();
        if (remoteUrl != null && !remoteUrl.trim().isEmpty()) {
            return createRemote(remoteUrl.trim());
        }
        return createLocal();
    }

    private static WebDriver createRemote(String remoteUrl) {
        try {
            MutableCapabilities caps = capabilities();
            RemoteWebDriver driver = new RemoteWebDriver(new URL(remoteUrl), caps);
            configureDriver(driver);
            return driver;
        } catch (Exception ex) {
            throw new IllegalStateException("No se pudo crear RemoteWebDriver. remoteUrl=" + remoteUrl, ex);
        }
    }

    private static WebDriver createLocal() {
        String browser = TestConfig.browser().toLowerCase();
        boolean headless = TestConfig.headless();

        switch (browser) {
            case "firefox":
                WebDriverManager.firefoxdriver().setup();
                FirefoxOptions ff = new FirefoxOptions();
                if (headless) {
                    ff.addArguments("-headless");
                }
                WebDriver ffDriver = new org.openqa.selenium.firefox.FirefoxDriver(ff);
                configureDriver(ffDriver);
                return ffDriver;
            case "chrome":
            default:
                WebDriverManager.chromedriver().setup();
                ChromeOptions ch = new ChromeOptions();
                if (headless) {
                    ch.addArguments("--headless=new");
                }
                ch.addArguments("--no-sandbox");
                ch.addArguments("--disable-dev-shm-usage");
                ch.addArguments("--window-size=1440,900");
                WebDriver chDriver = new org.openqa.selenium.chrome.ChromeDriver(ch);
                configureDriver(chDriver);
                return chDriver;
        }
    }

    private static MutableCapabilities capabilities() {
        String browser = TestConfig.browser().toLowerCase();
        boolean headless = TestConfig.headless();

        if ("firefox".equals(browser)) {
            FirefoxOptions options = new FirefoxOptions();
            options.setPlatformName(Platform.ANY.name());
            if (headless) {
                options.addArguments("-headless");
            }
            return options;
        }

        ChromeOptions options = new ChromeOptions();
        options.setPlatformName(Platform.ANY.name());
        if (headless) {
            options.addArguments("--headless=new");
        }
        options.addArguments("--no-sandbox");
        options.addArguments("--disable-dev-shm-usage");
        options.addArguments("--window-size=1440,900");
        return options;
    }

    private static void configureDriver(WebDriver driver) {
        driver.manage().timeouts().pageLoadTimeout(Duration.ofSeconds(TestConfig.timeoutSeconds()));
        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(0));
    }
}
