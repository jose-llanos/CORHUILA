package com.corhuila.gestionpruebas.selenium.utils;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.OutputType;
import org.openqa.selenium.TakesScreenshot;
import org.apache.commons.io.FileUtils;
import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class ScreenshotUtil {

    private static final String SCREENSHOT_DIR = "target/screenshots/";

    public static void tomarScreenshot(WebDriver driver, String nombre) {
        try {
            File screenshotDir = new File(SCREENSHOT_DIR);
            if (!screenshotDir.exists()) {
                screenshotDir.mkdirs();
            }

            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
            String fileName = String.format("%s_%s.png", nombre, timestamp);
            File destFile = new File(screenshotDir, fileName);

            TakesScreenshot ts = (TakesScreenshot) driver;
            File srcFile = ts.getScreenshotAs(OutputType.FILE);
            FileUtils.copyFile(srcFile, destFile);

            System.out.println("📸 Screenshot guardado: " + destFile.getAbsolutePath());
        } catch (IOException e) {
            System.err.println("Error al guardar screenshot: " + e.getMessage());
        }
    }
}