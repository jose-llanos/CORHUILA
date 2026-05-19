package edu.calidadsoftware.taskmanager.web;

import edu.calidadsoftware.taskmanager.task.Task;
import edu.calidadsoftware.taskmanager.task.TaskPriority;
import edu.calidadsoftware.taskmanager.task.TaskService;
import edu.calidadsoftware.taskmanager.task.TaskStatus;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.security.core.Authentication;
import org.springframework.ui.Model;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Pruebas unitarias para DashboardController.
 */
@DisplayName("DashboardController")
class DashboardControllerTest {

    @Test
    @DisplayName("Dashboard agrega totales al modelo")
    void dashboard_counts() {
        TaskService service = Mockito.mock(TaskService.class);
        when(service.findAll()).thenReturn(Arrays.asList(
                Task.builder().id(1L).title("a").status(TaskStatus.PENDING).priority(TaskPriority.LOW).build(),
                Task.builder().id(2L).title("b").status(TaskStatus.COMPLETED).priority(TaskPriority.HIGH).build()
        ));

        DashboardController controller = new DashboardController(service);
        Model model = Mockito.mock(Model.class);
        Authentication authentication = Mockito.mock(Authentication.class);
        when(authentication.getName()).thenReturn("admin");

        String view = controller.dashboard(model, authentication);

        assertEquals("dashboard", view);
        verify(model).addAttribute("totalTasks", 2);
        verify(model).addAttribute(Mockito.eq("counts"), Mockito.any());
        verify(model).addAttribute(Mockito.eq("statuses"), Mockito.any());
        verify(model).addAttribute("username", "admin");
    }
}
