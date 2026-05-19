package com.map.parking.selenium.tests;

import com.map.parking.selenium.config.WebDriverFactory;
import com.map.parking.selenium.support.TestUserSetup;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.openqa.selenium.WebDriver;

abstract class BaseUiTest {

    protected WebDriver driver;

    @BeforeAll
    static void prepareTestData() {
        TestUserSetup.ensureAdminUserExists();
    }

    @BeforeEach
    void setUpDriver() {
        driver = WebDriverFactory.createDriver();
    }

    @AfterEach
    void tearDownDriver() {
        if (driver != null) {
            driver.quit();
        }
    }
}
