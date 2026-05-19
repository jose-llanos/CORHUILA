package edu.calidadsoftware.taskmanager.selenium.tests;

import com.aventstack.extentreports.ExtentTest;
import edu.calidadsoftware.taskmanager.selenium.driver.DriverFactory;
import edu.calidadsoftware.taskmanager.selenium.driver.DriverManager;
import edu.calidadsoftware.taskmanager.selenium.report.ExtentTestWatcher;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.openqa.selenium.WebDriver;

/**
 * Base para pruebas UI.
 *
 * - Crea y destruye el WebDriver por prueba.
 * - Registra resultados y evidencias en ExtentReports.
 */
public abstract class BaseUiTest {

    @RegisterExtension
    static ExtentTestWatcher extent = new ExtentTestWatcher();

    @BeforeEach
    void setUpDriver() {
        WebDriver driver = DriverFactory.createDriver();
        DriverManager.set(driver);
    }

    @AfterEach
    void tearDownDriver() {
        WebDriver driver = DriverManager.get();
        try {
            if (driver != null) {
                driver.quit();
            }
        } finally {
            DriverManager.clear();
        }
    }

    protected ExtentTest report() {
        return ExtentTestWatcher.current();
    }
}

