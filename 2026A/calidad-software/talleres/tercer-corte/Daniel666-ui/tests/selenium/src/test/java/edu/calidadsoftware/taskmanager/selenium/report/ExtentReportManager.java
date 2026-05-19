package edu.calidadsoftware.taskmanager.selenium.report;

import com.aventstack.extentreports.ExtentReports;
import com.aventstack.extentreports.reporter.ExtentSparkReporter;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Inicializa ExtentReports (reporte HTML).
 *
 * El reporte se genera en /reports/entrega-2/selenium/extent-report.html (raíz del repo).
 */
public final class ExtentReportManager {

    private static ExtentReports instance;

    private ExtentReportManager() {
    }

    public static synchronized ExtentReports get() {
        if (instance == null) {
            instance = create();
        }
        return instance;
    }

    private static ExtentReports create() {
        try {
            Path base = Path.of(System.getProperty("user.dir"));
            if (base.endsWith(Path.of("tests", "selenium"))) {
                base = base.getParent().getParent();
            }

            Path reportDir = base.resolve(Path.of("reports", "entrega-2", "selenium"));
            Files.createDirectories(reportDir);
            Path reportFile = reportDir.resolve("extent-report.html");

            ExtentSparkReporter reporter = new ExtentSparkReporter(reportFile.toString());
            reporter.config().setDocumentTitle("Task Manager - Selenium Report");
            reporter.config().setReportName("Selenium Functional Tests");

            ExtentReports reports = new ExtentReports();
            reports.attachReporter(reporter);
            reports.setSystemInfo("Project", "Task Manager");
            reports.setSystemInfo("Course", "Calidad de Software 2026A");
            return reports;
        } catch (Exception ex) {
            throw new IllegalStateException("No se pudo inicializar ExtentReports", ex);
        }
    }
}
