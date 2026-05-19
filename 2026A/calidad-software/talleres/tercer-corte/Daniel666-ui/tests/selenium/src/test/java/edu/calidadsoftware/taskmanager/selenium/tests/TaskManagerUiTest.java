package edu.calidadsoftware.taskmanager.selenium.tests;

import edu.calidadsoftware.taskmanager.selenium.driver.DriverManager;
import edu.calidadsoftware.taskmanager.selenium.pages.DashboardPage;
import edu.calidadsoftware.taskmanager.selenium.pages.LoginPage;
import edu.calidadsoftware.taskmanager.selenium.pages.TaskFormPage;
import edu.calidadsoftware.taskmanager.selenium.pages.TaskListPage;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Casos funcionales (mínimo 6) según especificación del proyecto.
 *
 * Nota: Estas pruebas asumen que la aplicación está en ejecución en baseUrl (por defecto http://localhost:8080).
 */
@DisplayName("Task Manager - Selenium UI")
class TaskManagerUiTest extends BaseUiTest {

    @Test
    @DisplayName("TC01: Login exitoso con credenciales válidas")
    void tc01_loginSuccess() {
        report().info("Abrir login y autenticar con admin/admin");
        DashboardPage dashboard = new LoginPage().open().loginValid("admin", "admin");
        String url = DriverManager.get().getCurrentUrl();
        report().info("URL actual: " + url);
        assertTrue(url.contains("/dashboard") || url.endsWith("/"), "Debe redirigir al dashboard");
        dashboard.waitUntilLoaded();
    }

    @Test
    @DisplayName("TC02: Login fallido con credenciales inválidas (validar mensaje de error)")
    void tc02_loginFailure() {
        report().info("Intentar login con credenciales inválidas");
        LoginPage login = new LoginPage().open().loginInvalid("admin", "wrong");
        String message = login.getErrorMessage();
        report().info("Mensaje mostrado: " + message);
        assertTrue(message.contains("Credenciales inválidas"), "Debe mostrar mensaje de credenciales inválidas");
    }

    @Test
    @DisplayName("TC03: Crear nueva tarea y verificar que aparece en la lista")
    void tc03_createTask() {
        DashboardPage dashboard = new LoginPage().open().loginValid("admin", "admin");
        TaskListPage list = dashboard.goToTasks();

        String title = "Task Selenium " + Instant.now().toEpochMilli();
        report().info("Crear tarea: " + title);

        TaskFormPage form = list.clickNewTask();
        list = form.setTitle(title)
                .setDescription("Created by Selenium")
                .setStatus("PENDING")
                .setPriority("MEDIUM")
                .submit();

        report().info("Verificar que la tarea aparece en la lista");
        assertTrue(list.containsTitle(title), "La tarea creada debe aparecer en la lista");
    }

    @Test
    @DisplayName("TC04: Editar una tarea existente y confirmar cambios")
    void tc04_editTask() {
        DashboardPage dashboard = new LoginPage().open().loginValid("admin", "admin");
        TaskListPage list = dashboard.goToTasks();

        String originalTitle = "To edit " + Instant.now().toEpochMilli();
        String newTitle = originalTitle + " (edited)";

        report().info("Crear tarea base para edición: " + originalTitle);
        list = list.clickNewTask()
                .setTitle(originalTitle)
                .setDescription("Edit me")
                .setStatus("PENDING")
                .setPriority("LOW")
                .submit();

        report().info("Editar tarea y cambiar título a: " + newTitle);
        list = list.clickEditByTitle(originalTitle)
                .setTitle(newTitle)
                .setDescription("Edited by Selenium")
                .setStatus("IN_PROGRESS")
                .setPriority("HIGH")
                .submit();

        assertTrue(list.containsTitle(newTitle), "Debe aparecer el título actualizado");
        assertFalse(list.containsTitle(originalTitle), "No debe permanecer el título anterior");
    }

    @Test
    @DisplayName("TC05: Eliminar una tarea y confirmar que desaparece")
    void tc05_deleteTask() {
        DashboardPage dashboard = new LoginPage().open().loginValid("admin", "admin");
        TaskListPage list = dashboard.goToTasks();

        String title = "To delete " + Instant.now().toEpochMilli();

        report().info("Crear tarea base para eliminación: " + title);
        list = list.clickNewTask()
                .setTitle(title)
                .setDescription("Delete me")
                .setStatus("PENDING")
                .setPriority("MEDIUM")
                .submit();

        assertTrue(list.containsTitle(title), "La tarea debe existir antes de eliminar");

        report().info("Eliminar tarea y aceptar confirmación");
        list.deleteByTitle(title);

        report().info("Verificar que la tarea ya no aparece");
        assertFalse(list.containsTitle(title), "La tarea debe desaparecer tras eliminar");
    }

    @Test
    @DisplayName("TC06: Filtrar tareas por estado (pendiente/completada)")
    void tc06_filterByStatus() {
        DashboardPage dashboard = new LoginPage().open().loginValid("admin", "admin");
        TaskListPage list = dashboard.goToTasks();

        String pendingTitle = "Pending " + Instant.now().toEpochMilli();
        String completedTitle = "Completed " + (Instant.now().toEpochMilli() + 1);

        report().info("Crear tarea PENDING: " + pendingTitle);
        list = list.clickNewTask()
                .setTitle(pendingTitle)
                .setDescription("Pending task")
                .setStatus("PENDING")
                .setPriority("LOW")
                .submit();

        report().info("Crear tarea COMPLETED: " + completedTitle);
        list = list.clickNewTask()
                .setTitle(completedTitle)
                .setDescription("Completed task")
                .setStatus("COMPLETED")
                .setPriority("MEDIUM")
                .submit();

        report().info("Aplicar filtro COMPLETED");
        list.filterByStatus("COMPLETED");

        assertTrue(list.containsTitle(completedTitle), "La tarea COMPLETED debe mostrarse");
        assertFalse(list.containsTitle(pendingTitle), "La tarea PENDING no debe mostrarse con filtro COMPLETED");

        List<String> statuses = list.getDisplayedStatuses();
        report().info("Estados visibles: " + statuses);
        assertTrue(statuses.stream().allMatch(s -> s.equals("COMPLETED")), "Todas las filas visibles deben ser COMPLETED");
    }
}

