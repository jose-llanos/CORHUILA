package edu.calidadsoftware.taskmanager.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.support.ui.Select;

/**
 * Page Object: Formulario de tarea (crear/editar).
 */
public class TaskFormPage extends BasePage {

    private final By titleInput = By.id("title");
    private final By descriptionTextArea = By.id("description");
    private final By statusSelect = By.id("status");
    private final By prioritySelect = By.id("priority");
    private final By submitButton = By.cssSelector("button[type='submit']");

    public TaskFormPage waitUntilLoaded() {
        waitVisible(titleInput);
        return this;
    }

    public TaskFormPage setTitle(String title) {
        type(titleInput, title);
        return this;
    }

    public TaskFormPage setDescription(String description) {
        type(descriptionTextArea, description);
        return this;
    }

    public TaskFormPage setStatus(String status) {
        new Select(waitVisible(statusSelect)).selectByVisibleText(status);
        return this;
    }

    public TaskFormPage setPriority(String priority) {
        new Select(waitVisible(prioritySelect)).selectByVisibleText(priority);
        return this;
    }

    public TaskListPage submit() {
        waitClickable(submitButton).click();
        waitUrlContains("/tasks");
        waitDocumentReady();
        return new TaskListPage().waitUntilLoaded();
    }
}
