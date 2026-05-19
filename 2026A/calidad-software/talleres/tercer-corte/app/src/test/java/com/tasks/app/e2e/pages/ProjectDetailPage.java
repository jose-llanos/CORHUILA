package com.tasks.app.e2e.pages;
 
import org.openqa.selenium.By;
import org.openqa.selenium.StaleElementReferenceException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.Select;
import org.openqa.selenium.support.ui.WebDriverWait;
 
import java.time.Duration;
import java.util.List;
 
public class ProjectDetailPage extends BasePage {
 
    public enum Column {
        PENDING("col-pending", "PENDING"),
        IN_PROGRESS("col-in-progress", "IN_PROGRESS"),
        DONE("col-done", "DONE");
 
        public final String dataTest;
        public final String statusValue;
 
        Column(String dataTest, String statusValue) {
            this.dataTest = dataTest;
            this.statusValue = statusValue;
        }
    }
 
    public ProjectDetailPage(WebDriver driver, WebDriverWait wait, String baseUrl) {
        super(driver, wait, baseUrl);
        waitVisible(byTest("project-detail-view"));
    }
 
    public String getTitle() {
        return text(byTest("project-title"));
    }
 
    public String getDescription() {
        return text(byTest("project-description"));
    }
 
    // ---------- Proyecto ----------
 
    public GenericModalPage openEditProjectModal() {
        click(byTest("btn-edit-project"));
        return new GenericModalPage(driver, wait, baseUrl);
    }
 
    public DashboardPage deleteProject() {
        click(byTest("btn-delete-project"));
        try {
            wait.until(ExpectedConditions.alertIsPresent()).accept();
        } catch (org.openqa.selenium.TimeoutException ignored) {
        }
        waitVisible(byTest("empty-view"));
        return new DashboardPage(driver, wait, baseUrl);
    }
 
    // ---------- Tareas ----------
 
    public GenericModalPage openCreateTaskModal() {
        click(byTest("btn-new-task"));
        return new GenericModalPage(driver, wait, baseUrl);
    }
 
    public boolean isTaskInColumn(String taskTitle, Column column) {
        WebElement col = waitVisible(byTest(column.dataTest));
        List<WebElement> cards = col.findElements(
                By.cssSelector("[data-test^='task-card-']"));
        return cards.stream().anyMatch(c -> c.getText().contains(taskTitle));
    }
 
    public String findTaskIdByTitle(String title) {
        List<WebElement> cards = driver.findElements(
                By.cssSelector("[data-test^='task-card-']"));
        for (WebElement card : cards) {
            if (card.getText().contains(title)) {
                String dt = card.getAttribute("data-test");
                return dt.substring("task-card-".length());
            }
        }
        throw new AssertionError("No se encontró tarea con título: " + title);
    }
 
    public void changeTaskStatus(String taskId, Column newStatus) {
        WebElement select = waitVisible(byTest("select-status-" + taskId));
        new Select(select).selectByValue(newStatus.statusValue);
        wait.until(d -> isTaskInColumn(d.findElement(
                byTest("task-card-" + taskId)).getText(), newStatus));
    }
 
    public void assignTask(String taskId, String username) {
        WebElement select = waitVisible(byTest("select-assign-" + taskId));
        new Select(select).selectByVisibleText(username);
    }
 
    /**
     * Lee el assignee del select de la tarea.
     *
     * <p>Resiliente a re-renders del frontend: si el {@code <select>} o sus
     * {@code <option>} son reemplazados entre que los buscamos y los leemos
     * (StaleElementReferenceException), reintenta hasta 5 s. Esto ocurre
     * típicamente tras invitar/remover miembros, porque el front re-arma
     * el dropdown de asignación con la nueva lista de usuarios.</p>
     */
    public String getTaskAssignee(String taskId) {
        WebDriverWait shortWait = new WebDriverWait(driver, Duration.ofSeconds(5));
        return shortWait.ignoring(StaleElementReferenceException.class).until(d -> {
            WebElement select = d.findElement(byTest("select-assign-" + taskId));
            return new Select(select).getFirstSelectedOption().getText();
        });
    }
 
    // ---------- Miembros ----------
 
    public ProjectDetailPage inviteMember(String username) {
        type(byTest("input-invite-username"), username);
        click(byTest("btn-submit-invite"));
        // El front puede mostrar un alert ("Invitación enviada"); acéptalo si aparece.
        acceptAlertIfPresent();
        wait.until(d -> d.findElement(byTest("members-list"))
                .getText().contains(username));
        return this;
    }
 
    public boolean isMember(String username) {
        WebElement list = driver.findElement(byTest("members-list"));
        return list.getText().contains(username);
    }
 
    public ProjectDetailPage removeMember(String memberId) {
        click(byTest("btn-remove-member-" + memberId));
        try {
            wait.until(ExpectedConditions.alertIsPresent()).accept();
        } catch (org.openqa.selenium.TimeoutException ignored) {}
        waitInvisible(byTest("member-" + memberId));
        return this;
    }
 
    public String findMemberIdByUsername(String username) {
        List<WebElement> members = driver.findElements(
                By.cssSelector("[data-test^='member-']"));
        for (WebElement m : members) {
            if (m.getText().contains(username)) {
                String dt = m.getAttribute("data-test");
                return dt.substring("member-".length());
            }
        }
        throw new AssertionError("No se encontró miembro: " + username);
    }
 
    private void acceptAlertIfPresent() {
        try {
            wait.until(ExpectedConditions.alertIsPresent()).accept();
        } catch (org.openqa.selenium.TimeoutException ignored) {
        }
    }
}