package com.corhuila.gestionpruebas.controller;

import com.corhuila.gestionpruebas.model.Duenio;
import com.corhuila.gestionpruebas.service.DuenioService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

@Controller
@RequestMapping("/duenios")
public class DuenioController {

    @Autowired
    private DuenioService duenioService;

    @GetMapping
    public String listar(Model model) {
        model.addAttribute("listaDuenios", duenioService.obtenerTodos());
        return "duenios/lista";
    }

    @GetMapping("/nuevo")
    public String formulario(Model model) {
        model.addAttribute("duenio", new Duenio());
        return "duenios/formulario";
    }

    @PostMapping("/guardar")
    public String guardar(@ModelAttribute Duenio duenio) {
        duenioService.guardar(duenio);
        return "redirect:/duenios";
    }

    @GetMapping("/{id}/editar")
    public String editar(@PathVariable Long id, Model model) {
        model.addAttribute("duenio", duenioService.buscarPorId(id));
        return "duenios/formulario";
    }

    @PostMapping("/{id}/eliminar")
    public String eliminar(@PathVariable Long id, RedirectAttributes redirectAttributes) {
        try {
            duenioService.eliminar(id);
            redirectAttributes.addFlashAttribute("mensaje", "Dueño eliminado correctamente.");
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error",
                    "No se puede eliminar este dueño porque tiene mascotas registradas. " +
                            "Elimine primero las mascotas asociadas.");
        }
        return "redirect:/duenios";
    }
}