package com.corhuila.gestionpruebas.controller;

import com.corhuila.gestionpruebas.model.Tratamiento;
import com.corhuila.gestionpruebas.service.CitaService;
import com.corhuila.gestionpruebas.service.TratamientoService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping("/tratamientos")
public class TratamientoController {

    @Autowired private TratamientoService tratamientoService;
    @Autowired private CitaService citaService;

    @GetMapping
    public String listar(Model model) {
        model.addAttribute("listaTratamientos", tratamientoService.obtenerTodos());
        return "tratamientos/lista";
    }

    @GetMapping("/nuevo")
    public String formulario(Model model) {
        model.addAttribute("tratamiento", new Tratamiento());
        model.addAttribute("listaCitas", citaService.obtenerTodas());
        return "tratamientos/formulario";
    }

    @PostMapping("/guardar")
    public String guardar(@ModelAttribute Tratamiento tratamiento) {
        tratamientoService.guardar(tratamiento);
        return "redirect:/tratamientos";
    }

    @GetMapping("/{id}/editar")
    public String editar(@PathVariable Long id, Model model) {
        model.addAttribute("tratamiento", tratamientoService.buscarPorId(id));
        model.addAttribute("listaCitas", citaService.obtenerTodas());
        return "tratamientos/formulario";
    }

    @PostMapping("/{id}/eliminar")
    public String eliminar(@PathVariable Long id) {
        tratamientoService.eliminar(id);
        return "redirect:/tratamientos";
    }
}