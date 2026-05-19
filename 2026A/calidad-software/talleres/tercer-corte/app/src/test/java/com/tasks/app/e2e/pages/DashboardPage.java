package com.tasks.app.e2e.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.util.List;

public class DashboardPage extends BasePage {

    public DashboardPage(WebDriver driver, WebDriverWait wait, String baseUrl) {
        super(driver, wait, baseUrl);
        waitVisible(byTest("username-display"));
    }

    public String getDisplayedUsername() {
        return text(byTest("username-display"));
    }

    public GenericModalPage openCreateProjectModal() {
        click(byTest("btn-new-project"));
        return new GenericModalPage(driver, wait, baseUrl);
    }

    /**
     * Selecciona un proyecto por su nombre visible. Espera a que el
     * detalle se renderice (project-detail-view visible).
     */
    public ProjectDetailPage selectProjectByName(String name) {
        By container = byTest("projects-list-container");
        waitVisible(container);
        // Cada item tiene data-test="project-item-{id}"; filtramos por texto.
        List<WebElement> items = driver.findElements(
                By.cssSelector("[data-test^='project-item-']"));
        WebElement target = items.stream()
                .filter(e -> e.getText().contains(name))
                .findFirst()
                .orElseThrow(() -> new AssertionError(
                        "No se encontró el proyecto: " + name));
        target.click();
        waitVisible(byTest("project-detail-view"));
        return new ProjectDetailPage(driver, wait, baseUrl);
    }

    public boolean projectExists(String name) {
        List<WebElement> items = driver.findElements(
                By.cssSelector("[data-test^='project-item-']"));
        return items.stream().anyMatch(e -> e.getText().contains(name));
    }

    public ProfileModalPage openProfileModal() {
        click(byTest("username-display"));
        return new ProfileModalPage(driver, wait, baseUrl);
    }

    public LoginPage logout() {
        click(byTest("btn-logout"));
        wait.until(ExpectedConditions.urlContains("index.html"));
        return new LoginPage(driver, wait, baseUrl);
    }
}
