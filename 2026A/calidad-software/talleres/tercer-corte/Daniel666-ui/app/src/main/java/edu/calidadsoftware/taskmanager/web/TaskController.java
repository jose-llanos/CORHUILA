package edu.calidadsoftware.taskmanager.web;

import edu.calidadsoftware.taskmanager.task.Task;
import edu.calidadsoftware.taskmanager.task.TaskForm;
import edu.calidadsoftware.taskmanager.task.TaskPriority;
import edu.calidadsoftware.taskmanager.task.TaskService;
import edu.calidadsoftware.taskmanager.task.TaskStatus;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import javax.validation.Valid;
import java.util.List;

/**
 * Controlador MVC para CRUD de tareas (UI con Thymeleaf).
 */
@Controller
@RequestMapping("/tasks")
@RequiredArgsConstructor
public class TaskController {

    private final TaskService taskService;

    @GetMapping
    public String list(@RequestParam(value = "status", required = false) TaskStatus status, Model model) {
        List<Task> tasks = (status == null) ? taskService.findAll() : taskService.findByStatus(status);
        model.addAttribute("tasks", tasks);
        model.addAttribute("selectedStatus", status);
        model.addAttribute("statuses", TaskStatus.values());
        return "tasks/list";
    }

    @GetMapping("/new")
    public String createForm(Model model) {
        model.addAttribute("taskForm", TaskForm.builder()
                .status(TaskStatus.PENDING)
                .priority(TaskPriority.MEDIUM)
                .build());
        model.addAttribute("statuses", TaskStatus.values());
        model.addAttribute("priorities", TaskPriority.values());
        model.addAttribute("mode", "create");
        return "tasks/form";
    }

    @PostMapping
    public String create(@Valid TaskForm taskForm, BindingResult bindingResult, Model model, RedirectAttributes redirectAttributes) {
        if (bindingResult.hasErrors()) {
            model.addAttribute("statuses", TaskStatus.values());
            model.addAttribute("priorities", TaskPriority.values());
            model.addAttribute("mode", "create");
            return "tasks/form";
        }
        taskService.createTask(taskForm);
        redirectAttributes.addFlashAttribute("message", "Task created successfully");
        return "redirect:/tasks";
    }

    @GetMapping("/{id}/edit")
    public String editForm(@PathVariable Long id, Model model) {
        Task task = taskService.findById(id);
        model.addAttribute("taskId", id);
        model.addAttribute("taskForm", TaskForm.builder()
                .title(task.getTitle())
                .description(task.getDescription())
                .status(task.getStatus())
                .priority(task.getPriority())
                .build());
        model.addAttribute("statuses", TaskStatus.values());
        model.addAttribute("priorities", TaskPriority.values());
        model.addAttribute("mode", "edit");
        return "tasks/form";
    }

    @PostMapping("/{id}")
    public String update(@PathVariable Long id, @Valid TaskForm taskForm, BindingResult bindingResult, Model model, RedirectAttributes redirectAttributes) {
        if (bindingResult.hasErrors()) {
            model.addAttribute("taskId", id);
            model.addAttribute("statuses", TaskStatus.values());
            model.addAttribute("priorities", TaskPriority.values());
            model.addAttribute("mode", "edit");
            return "tasks/form";
        }
        taskService.updateTask(id, taskForm);
        redirectAttributes.addFlashAttribute("message", "Task updated successfully");
        return "redirect:/tasks";
    }

    @PostMapping("/{id}/delete")
    public String delete(@PathVariable Long id, RedirectAttributes redirectAttributes) {
        taskService.deleteTask(id);
        redirectAttributes.addFlashAttribute("message", "Task deleted successfully");
        return "redirect:/tasks";
    }
}
