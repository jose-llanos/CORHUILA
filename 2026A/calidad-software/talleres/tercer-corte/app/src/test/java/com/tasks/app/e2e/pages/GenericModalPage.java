package com.tasks.app.e2e.pages;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.support.ui.WebDriverWait;

/**
 * Modal genérico que se reutiliza para crear/editar proyectos y tareas.
 * Los selectores varían según el contexto (project-name vs task-title);
 * exponemos métodos específicos por flujo.
 */
public class GenericModalPage extends BasePage {

    public GenericModalPage(WebDriver driver, WebDriverWait wait, String baseUrl) {
        super(driver, wait, baseUrl);
        waitVisible(byTest("modal-container"));
    }

    public GenericModalPage fillProject(String name, String description) {
        type(byTest("modal-input-project-name"), name);
        type(byTest("modal-input-project-desc"), description);
        return this;
    }

    public GenericModalPage fillTask(String title, String description) {
        type(byTest("modal-input-task-title"), title);
        type(byTest("modal-input-task-desc"), description);
        return this;
    }

    public void save() {
        click(byTest("modal-save-btn"));
        waitInvisible(byTest("modal-container"));
    }
}
