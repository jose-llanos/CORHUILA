package edu.calidadsoftware.taskmanager.web;

import edu.calidadsoftware.taskmanager.task.Task;
import edu.calidadsoftware.taskmanager.task.TaskService;
import edu.calidadsoftware.taskmanager.task.TaskStatus;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Dashboard básico del sistema.
 *
 * Muestra indicadores simples (cantidad de tareas por estado) y accesos a funcionalidades.
 */
@Controller
@RequiredArgsConstructor
public class DashboardController {

    private final TaskService taskService;

    @GetMapping({"/", "/dashboard"})
    public String dashboard(Model model, Authentication authentication) {
        List<Task> tasks = taskService.findAll();

        Map<String, Long> counts = new LinkedHashMap<>();
        for (TaskStatus status : TaskStatus.values()) {
            counts.put(status.name(), 0L);
        }
        tasks.forEach(task -> {
            TaskStatus status = task.getStatus();
            if (status != null) {
                String key = status.name();
                counts.put(key, counts.get(key) + 1);
            }
        });

        model.addAttribute("totalTasks", tasks.size());
        model.addAttribute("counts", counts);
        model.addAttribute("statuses", TaskStatus.values());
        model.addAttribute("username", authentication != null ? authentication.getName() : "guest");
        return "dashboard";
    }
}
