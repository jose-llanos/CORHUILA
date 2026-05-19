package com.tasks.app.e2e.config;
 
import org.junit.jupiter.api.extension.AfterTestExecutionCallback;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.openqa.selenium.OutputType;
import org.openqa.selenium.TakesScreenshot;
import org.openqa.selenium.WebDriver;
 
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
 
/**
 * Captura un screenshot en {@code target/screenshots/} si el test falla.
 * El driver se obtiene de un {@link ThreadLocal} poblado por {@link BaseE2ETest}.
 */
public class ScreenshotOnFailure implements AfterTestExecutionCallback {
 
    private static final Path OUTPUT_DIR = Paths.get("target", "screenshots");
    private static final DateTimeFormatter STAMP =
            DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss-SSS");
 
    @Override
    public void afterTestExecution(ExtensionContext ctx) {
        boolean failed = ctx.getExecutionException().isPresent();
        if (!failed) return;
 
        WebDriver driver = BaseE2ETest.currentDriver();
        if (driver == null) return;
 
        try {
            Files.createDirectories(OUTPUT_DIR);
            byte[] bytes = ((TakesScreenshot) driver).getScreenshotAs(OutputType.BYTES);
            String name = ctx.getRequiredTestClass().getSimpleName()
                    + "_" + ctx.getRequiredTestMethod().getName()
                    + "_" + LocalDateTime.now().format(STAMP)
                    + ".png";
            Files.write(OUTPUT_DIR.resolve(name), bytes);
            System.out.println("[E2E] Screenshot guardado: " + OUTPUT_DIR.resolve(name));
        } catch (IOException e) {
            System.err.println("[E2E] No se pudo guardar el screenshot: " + e.getMessage());
        }
    }
}