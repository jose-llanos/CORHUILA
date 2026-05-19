package edu.calidadsoftware.taskmanager.selenium.pages;

import edu.calidadsoftware.taskmanager.selenium.config.TestConfig;
import org.openqa.selenium.By;

/**
 * Page Object: Dashboard.
 */
public class DashboardPage extends BasePage {

    private final By dashboardTitle = By.cssSelector("h1.title");
    private final By tasksLink = By.cssSelector("a[href='/tasks']");

    public DashboardPage open() {
        driver.get(TestConfig.baseUrl() + "/dashboard");
        return waitUntilLoaded();
    }

    public DashboardPage waitUntilLoaded() {
        waitVisible(dashboardTitle);
        return this;
    }

    public TaskListPage goToTasks() {
        driver.get(TestConfig.baseUrl() + "/tasks");
        return new TaskListPage().waitUntilLoaded();
    }
}
