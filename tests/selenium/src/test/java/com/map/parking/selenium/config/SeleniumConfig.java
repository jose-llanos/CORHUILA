package com.map.parking.selenium.config;

public final class SeleniumConfig {

    public static final String BASE_URL = System.getProperty("base.url", "http://localhost:4200");
    public static final String API_URL = System.getProperty("api.url", "http://localhost:8080");
    public static final String BROWSER = System.getProperty("browser", "brave");
    public static final int TIMEOUT_SECONDS = Integer.parseInt(System.getProperty("timeout.seconds", "15"));

    public static final String TEST_ADMIN_EMAIL = "selenium.admin@test.com";
    public static final String TEST_ADMIN_PASSWORD = "Test1234!";

    private SeleniumConfig() {
    }
}
