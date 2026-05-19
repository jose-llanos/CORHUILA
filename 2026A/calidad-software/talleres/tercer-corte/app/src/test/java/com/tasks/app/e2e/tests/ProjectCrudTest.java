package com.tasks.app.e2e.tests;
 
import com.tasks.app.e2e.config.BaseE2ETest;
import com.tasks.app.e2e.config.BrowserType;
import com.tasks.app.e2e.pages.DashboardPage;
import com.tasks.app.e2e.pages.LoginPage;
import com.tasks.app.e2e.pages.ProjectDetailPage;
import com.tasks.app.e2e.utils.TestDataFactory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
 
import static org.assertj.core.api.Assertions.assertThat;
 
/**
 * Caso 2 (TC-PROJ-001) — RF-02.1, RF-02.2, RF-02.4, RF-02.5.
 */
class ProjectCrudTest extends BaseE2ETest {
 
    @ParameterizedTest(name = "CRUD de proyecto en {0}")
    @EnumSource(BrowserType.class)
    void shouldCreateEditAndDeleteProject(BrowserType browser) {
        setupDriver(browser);
 
        // Setup: usuario nuevo
        String user = TestDataFactory.uniqueUsername();
        String pass = TestDataFactory.DEFAULT_PASSWORD;
        new LoginPage(driver, wait, baseUrl).open()
                .goToRegister()
                .register(user, TestDataFactory.emailFor(user), pass);
 
        DashboardPage dashboard = new LoginPage(driver, wait, baseUrl).open()
                .login(user, pass);
 
        // 1) Crear
        String name = TestDataFactory.uniqueProjectName();
        dashboard.openCreateProjectModal()
                .fillProject(name, "Descripción inicial")
                .save();
        assertThat(dashboard.projectExists(name)).isTrue();
 
        // 2) Editar
        ProjectDetailPage detail = dashboard.selectProjectByName(name);
        String renamed = name + "_v2";
        detail.openEditProjectModal()
                .fillProject(renamed, "Descripción editada")
                .save();
        assertThat(detail.getTitle()).contains(renamed);
        assertThat(detail.getDescription()).contains("editada");
 
        // 3) Eliminar
        DashboardPage afterDelete = detail.deleteProject();
        assertThat(afterDelete.projectExists(renamed)).isFalse();
    }
}