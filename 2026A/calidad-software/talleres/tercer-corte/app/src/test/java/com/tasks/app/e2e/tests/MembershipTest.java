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
 * Caso 4 (TC-MEMB-001) — RF-02.6, RF-02.7, RF-02.8, RF-03.5.
 */
class MembershipTest extends BaseE2ETest {
 
    @ParameterizedTest(name = "Invitar/asignar/remover miembro en {0}")
    @EnumSource(BrowserType.class)
    void shouldInviteAssignAndRemoveMember(BrowserType browser) {
        setupDriver(browser);
 
        String pass = TestDataFactory.DEFAULT_PASSWORD;
 
        // Usuario A (dueño) y usuario B (invitado), ambos registrados.
        String userA = TestDataFactory.uniqueUsername();
        String userB = TestDataFactory.uniqueUsername();
 
        // Registrar B primero (queda en la BD).
        new LoginPage(driver, wait, baseUrl).open()
                .goToRegister()
                .register(userB, TestDataFactory.emailFor(userB), pass);
 
        // Registrar A.
        new LoginPage(driver, wait, baseUrl).open()
                .goToRegister()
                .register(userA, TestDataFactory.emailFor(userA), pass);
 
        // Login como A y crear proyecto.
        DashboardPage dashboard = new LoginPage(driver, wait, baseUrl).open()
                .login(userA, pass);
 
        String project = TestDataFactory.uniqueProjectName();
        dashboard.openCreateProjectModal()
                .fillProject(project, "proyecto compartido")
                .save();
        ProjectDetailPage detail = dashboard.selectProjectByName(project);
 
        // 1) A invita a B
        detail.inviteMember(userB);
        assertThat(detail.isMember(userB)).isTrue();
 
        // 2) A crea una tarea y se la asigna a B
        String task = TestDataFactory.uniqueTaskTitle();
        detail.openCreateTaskModal().fillTask(task, "tarea asignada").save();
        String taskId = detail.findTaskIdByTitle(task);
        detail.assignTask(taskId, userB);
        assertThat(detail.getTaskAssignee(taskId)).isEqualTo(userB);
        assertThat(detail.isTaskInColumn(task, Column.PENDING)).isTrue();
 
        // 3) A remueve a B; la tarea debe quedar sin asignar
        String memberId = detail.findMemberIdByUsername(userB);
        detail.removeMember(memberId);
        assertThat(detail.isMember(userB)).isFalse();
        // Tras remover, el assignee debería volver a un placeholder
        // (texto típico: "Sin asignar" o similar). Verificamos que ya no sea userB.
        assertThat(detail.getTaskAssignee(taskId)).isNotEqualTo(userB);
    }
}