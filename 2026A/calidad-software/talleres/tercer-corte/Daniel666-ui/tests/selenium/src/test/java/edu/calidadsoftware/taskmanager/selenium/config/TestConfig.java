package edu.calidadsoftware.taskmanager.selenium.config;

/**
 * Configuración de ejecución para pruebas Selenium.
 *
 * Se obtiene por orden de prioridad:
 * 1) System properties (-DbaseUrl, -Dbrowser, -Dheadless, -DremoteUrl)
 * 2) Variables de entorno (TASKMANAGER_BASE_URL, SELENIUM_BROWSER, SELENIUM_HEADLESS, SELENIUM_REMOTE_URL)
 * 3) Valores por defecto
 */
public final class TestConfig {

    private TestConfig() {
    }

    public static String baseUrl() {
        return read("baseUrl", "TASKMANAGER_BASE_URL", "http://localhost:8080");
    }

    public static String browser() {
        return read("browser", "SELENIUM_BROWSER", "chrome");
    }

    public static boolean headless() {
        return Boolean.parseBoolean(read("headless", "SELENIUM_HEADLESS", "true"));
    }

    public static String remoteUrl() {
        return read("remoteUrl", "SELENIUM_REMOTE_URL", "");
    }

    public static int timeoutSeconds() {
        String value = read("timeoutSeconds", "SELENIUM_TIMEOUT_SECONDS", "15");
        try {
            return Integer.parseInt(value);
        } catch (Exception ex) {
            return 15;
        }
    }

    private static String read(String propertyKey, String envKey, String defaultValue) {
        String prop = System.getProperty(propertyKey);
        if (prop != null && !prop.trim().isEmpty()) {
            return prop.trim();
        }
        String env = System.getenv(envKey);
        if (env != null && !env.trim().isEmpty()) {
            return env.trim();
        }
        return defaultValue;
    }
}
