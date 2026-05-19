package edu.calidadsoftware.taskmanager.selenium.report;

import com.aventstack.extentreports.ExtentReports;
import com.aventstack.extentreports.ExtentTest;
import edu.calidadsoftware.taskmanager.selenium.driver.DriverManager;
import org.junit.jupiter.api.extension.AfterAllCallback;
import org.junit.jupiter.api.extension.BeforeTestExecutionCallback;
import org.junit.jupiter.api.extension.BeforeAllCallback;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.extension.TestWatcher;
import org.openqa.selenium.OutputType;
import org.openqa.selenium.TakesScreenshot;
import org.openqa.selenium.WebDriver;

import java.util.Optional;

/**
 * Extensión JUnit 5:
 * - Crea un nodo de reporte por test.
 * - Adjunta captura de pantalla cuando la prueba falla.
 * - Hace flush del reporte al final.
 */
public class ExtentTestWatcher implements BeforeAllCallback, AfterAllCallback, BeforeTestExecutionCallback, TestWatcher {

    private static final ThreadLocal<ExtentTest> CURRENT_TEST = new ThreadLocal<>();

    @Override
    public void beforeAll(ExtensionContext context) {
        ExtentReportManager.get();
    }

    @Override
    public void beforeTestExecution(ExtensionContext context) {
        getOrCreateTest(context);
    }

    @Override
    public void afterAll(ExtensionContext context) {
        ExtentReports reports = ExtentReportManager.get();
        reports.flush();
    }

    @Override
    public void testSuccessful(ExtensionContext context) {
        ExtentTest test = getOrCreateTest(context);
        test.pass("Caso ejecutado correctamente.");
        CURRENT_TEST.remove();
    }

    @Override
    public void testFailed(ExtensionContext context, Throwable cause) {
        ExtentTest test = getOrCreateTest(context);
        test.fail(cause);

        WebDriver driver = DriverManager.get();
        if (driver instanceof TakesScreenshot) {
            try {
                String base64 = ((TakesScreenshot) driver).getScreenshotAs(OutputType.BASE64);
                test.addScreenCaptureFromBase64String(base64, "Evidencia de fallo");
            } catch (Exception ex) {
                test.warning("No se pudo capturar screenshot: " + ex.getMessage());
            }
        }
        CURRENT_TEST.remove();
    }

    @Override
    public void testAborted(ExtensionContext context, Throwable cause) {
        ExtentTest test = getOrCreateTest(context);
        test.skip("Caso abortado: " + cause.getMessage());
        CURRENT_TEST.remove();
    }

    @Override
    public void testDisabled(ExtensionContext context, Optional<String> reason) {
        ExtentTest test = getOrCreateTest(context);
        test.skip("Caso deshabilitado: " + reason.orElse("sin motivo"));
        CURRENT_TEST.remove();
    }

    public static ExtentTest current() {
        return CURRENT_TEST.get();
    }

    private ExtentTest getOrCreateTest(ExtensionContext context) {
        ExtentTest existing = CURRENT_TEST.get();
        if (existing != null) {
            return existing;
        }
        String name = context.getDisplayName();
        ExtentTest created = ExtentReportManager.get().createTest(name);
        CURRENT_TEST.set(created);
        return created;
    }
}
