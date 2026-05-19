package com.map.parking.selenium.config;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Local: Brave (Chromium). CI/GitHub Actions: Chrome headless.
 */
public final class WebDriverFactory {

    public static WebDriver createDriver() {
        return createDriver(SeleniumConfig.BROWSER);
    }

    public static WebDriver createDriver(String browser) {
        String normalized = browser == null ? "brave" : browser.trim().toLowerCase();
        return switch (normalized) {
            case "chrome" -> createChrome(null);
            case "brave"  -> createBrave();
            default       -> braveBinary().isPresent() ? createBrave() : createChrome(null);
        };
    }

    private static WebDriver createBrave() {
        Path binary = braveBinary().orElseThrow(() -> new IllegalStateException(
                "Brave no encontrado. Instala Brave o usa -Dbrowser=chrome"));
        System.out.println("[WebDriver] Usando Brave: " + binary);
        return createChrome(binary);
    }

    private static WebDriver createChrome(Path binary) {
        ChromeOptions options = new ChromeOptions();
        if (binary != null) {
            options.setBinary(binary.toString());
        }
        applyChromiumArgs(options);
        return new ChromeDriver(options);
    }

    private static void applyChromiumArgs(ChromeOptions options) {
        options.addArguments("--start-maximized");
        options.addArguments("--disable-search-engine-choice-screen");
        options.addArguments("--remote-allow-origins=*");
        options.addArguments("--no-sandbox");
        options.addArguments("--disable-dev-shm-usage");
        if (Boolean.parseBoolean(System.getProperty("headless", "false"))) {
            options.addArguments("--headless=new");
        }
    }

    private static Optional<Path> braveBinary() {
        List<Path> candidates = new ArrayList<>();

        // ── Windows ──────────────────────────────────────────────
        addWindowsPath(candidates, "ProgramFiles");
        addWindowsPath(candidates, "ProgramFiles(x86)");
        addWindowsPath(candidates, "LOCALAPPDATA");

        // ── Linux ─────────────────────────────────────────────────
        candidates.add(Path.of("/usr/bin/brave-browser"));
        candidates.add(Path.of("/usr/bin/brave"));
        candidates.add(Path.of("/snap/bin/brave"));

        // ── macOS ─────────────────────────────────────────────────
        candidates.add(Path.of("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"));

        return firstExisting(candidates);
    }

    /** Agrega la ruta de Windows solo si la variable de entorno existe. */
    private static void addWindowsPath(List<Path> list, String envVar) {
        String base = System.getenv(envVar);
        if (base != null && !base.isBlank()) {
            list.add(Path.of(base, "BraveSoftware", "Brave-Browser", "Application", "brave.exe"));
        }
    }

    private static Optional<Path> firstExisting(List<Path> paths) {
        return paths.stream()
                .filter(p -> p != null && Files.isRegularFile(p))
                .findFirst();
    }

    private WebDriverFactory() {}
}