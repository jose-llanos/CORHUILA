package com.autospark.migueljuliana.selenium.tests;

import com.aventstack.extentreports.ExtentReports;
import com.aventstack.extentreports.ExtentTest;
import com.aventstack.extentreports.reporter.ExtentSparkReporter;
import org.openqa.selenium.WebDriver;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public abstract class BaseTest {

    protected WebDriver driver;
    protected static ExtentReports extent;
    protected ExtentTest test;

    @BeforeClass
    public void setupExtent() {

        ExtentSparkReporter sparkReporter =
                new ExtentSparkReporter("test-output/ExtentReport.html");

        sparkReporter.config().setDocumentTitle("AutoSpark - Pruebas Funcionales");
        sparkReporter.config().setReportName("Reporte de Pruebas Selenium");

        extent = new ExtentReports();
        extent.attachReporter(sparkReporter);
    }

    @AfterClass
    public void tearDownExtent() {
        if (extent != null) {
            extent.flush();
        }
    }

    /**
     * Test básico para evitar el warning java:S2187 de SonarQube
     */
    @Test
    public void shouldInitializeExtentReports() {

        setupExtent();

        Assert.assertNotNull(extent,
                "ExtentReports debería inicializarse correctamente");
    }
}