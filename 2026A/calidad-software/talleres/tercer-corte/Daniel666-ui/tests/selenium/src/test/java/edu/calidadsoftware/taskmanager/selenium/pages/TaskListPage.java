package edu.calidadsoftware.taskmanager.selenium.pages;

import edu.calidadsoftware.taskmanager.selenium.config.TestConfig;
import org.openqa.selenium.By;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.Select;

import java.util.ArrayList;
import java.util.List;

/**
 * Page Object: Lista de tareas.
 */
public class TaskListPage extends BasePage {

    private final By pageTitle = By.cssSelector("h1.title");
    private final By newTaskButton = By.id("newTaskButton");
    private final By statusFilter = By.id("status");
    private final By table = By.id("taskTable");
    private final By flashMessage = By.cssSelector(".alert.alert-ok");

    public TaskListPage open() {
        driver.get(TestConfig.baseUrl() + "/tasks");
        return waitUntilLoaded();
    }

    public TaskListPage waitUntilLoaded() {
        waitVisible(pageTitle);
        waitUrlContains("/tasks");
        waitDocumentReady();
        waitVisible(table);
        return this;
    }

    public TaskFormPage clickNewTask() {
        waitClickable(newTaskButton).click();
        return new TaskFormPage().waitUntilLoaded();
    }

    public TaskListPage filterByStatus(String status) {
        WebElement selectEl = waitVisible(statusFilter);
        Select select = new Select(selectEl);
        if (status == null || status.trim().isEmpty()) {
            select.selectByVisibleText("All");
        } else {
            select.selectByVisibleText(status);
        }
        wait.until(ExpectedConditions.visibilityOfElementLocated(table));
        return this;
    }

    public boolean containsTitle(String title) {
        return findRowByTitle(title) != null;
    }

    public TaskFormPage clickEditByTitle(String title) {
        WebElement row = findRowByTitle(title);
        if (row == null) {
            throw new IllegalStateException("No se encontró fila para title=" + title);
        }
        row.findElement(By.cssSelector("a[data-testid='editTask']")).click();
        return new TaskFormPage().waitUntilLoaded();
    }

    public TaskListPage deleteByTitle(String title) {
        WebElement row = findRowByTitle(title);
        if (row == null) {
            throw new IllegalStateException("No se encontró fila para title=" + title);
        }
        row.findElement(By.cssSelector("button[data-testid='deleteTask']")).click();
        acceptConfirmIfPresent();
        wait.until(ExpectedConditions.visibilityOfElementLocated(table));
        return this;
    }

    public String waitFlashMessage() {
        return waitVisible(flashMessage).getText().trim();
    }

    public List<String> getDisplayedStatuses() {
        List<WebElement> rows = driver.findElements(By.cssSelector("#taskTable tbody tr[data-testid='taskRow']"));
        List<String> statuses = new ArrayList<>();
        for (WebElement row : rows) {
            List<WebElement> cols = row.findElements(By.tagName("td"));
            if (cols.size() >= 4) {
                statuses.add(cols.get(3).getText().trim());
            }
        }
        return statuses;
    }

    private WebElement findRowByTitle(String title) {
        List<WebElement> rows = driver.findElements(By.cssSelector("#taskTable tbody tr[data-testid='taskRow']"));
        for (WebElement row : rows) {
            List<WebElement> cols = row.findElements(By.tagName("td"));
            if (cols.size() >= 2) {
                String cellTitle = cols.get(1).getText().trim();
                if (cellTitle.equals(title)) {
                    return row;
                }
            }
        }
        return null;
    }

    private void acceptConfirmIfPresent() {
        try {
            wait.until(ExpectedConditions.alertIsPresent());
            driver.switchTo().alert().accept();
        } catch (Exception ex) {
        }
    }
}
