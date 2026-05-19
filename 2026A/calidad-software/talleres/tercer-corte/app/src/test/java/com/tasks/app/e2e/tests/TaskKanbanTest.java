package com.tasks.app.e2e.tests;
 
import com.tasks.app.e2e.config.BaseE2ETest;
import com.tasks.app.e2e.config.BrowserType;
import com.tasks.app.e2e.pages.DashboardPage;
import com.tasks.app.e2e.pages.LoginPage;
import com.tasks.app.e2e.pages.ProjectDetailPage;
import com.tasks.app.e2e.pages.ProjectDetailPage.Column;
import com.tasks.app.e2e.utils.TestDataFactory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
 
import static org.assertj.core.api.Assertions.assertThat;
 
/**
 * Caso 3 (TC-TASK-001) — RF-03.1, RF-03.4, RF-03.6.
 */
class TaskKanbanTest extends BaseE2ETest {
 
    @ParameterizedTest(name = "Crear tarea y mover por kanban en {0}")
    @EnumSource(BrowserType.class)
    void shouldCreateTaskAndMoveThroughKanban(BrowserType browser) {
        setupDriver(browser);
 
        // Setup: usuario + proyecto
        String user = TestDataFactory.uniqueUsername();
        String pass = TestDataFactory.DEFAULT_PASSWORD;
        new LoginPage(driver, wait, baseUrl).open()
                .goToRegister()
                .register(user, TestDataFactory.emailFor(user), pass);
 
        DashboardPage dashboard = new LoginPage(driver, wait, baseUrl).open()
                .login(user, pass);
 
        String project = TestDataFactory.uniqueProjectName();
        dashboard.openCreateProjectModal()
                .fillProject(project, "proyecto para kanban")
                .save();
        ProjectDetailPage detail = dashboard.selectProjectByName(project);
 
        // 1) Crear tarea: arranca en PENDING
        String task = TestDataFactory.uniqueTaskTitle();
        detail.openCreateTaskModal()
                .fillTask(task, "descripción de la tarea")
                .save();
        assertThat(detail.isTaskInColumn(task, Column.PENDING)).isTrue();
 
        // 2) Mover a IN_PROGRESS
        String taskId = detail.findTaskIdByTitle(task);
        detail.changeTaskStatus(taskId, Column.IN_PROGRESS);
        assertThat(detail.isTaskInColumn(task, Column.IN_PROGRESS)).isTrue();
        assertThat(detail.isTaskInColumn(task, Column.PENDING)).isFalse();
 
        // 3) Mover a DONE
        detail.changeTaskStatus(taskId, Column.DONE);
        assertThat(detail.isTaskInColumn(task, Column.DONE)).isTrue();
        assertThat(detail.isTaskInColumn(task, Column.IN_PROGRESS)).isFalse();
    }
}