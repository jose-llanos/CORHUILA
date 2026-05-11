package com.corhuila.gestionpruebas.controller;

import com.corhuila.gestionpruebas.model.Cita;
import com.corhuila.gestionpruebas.service.CitaService;
import com.corhuila.gestionpruebas.service.MascotaService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping("/citas")
public class CitaController {

    @Autowired private CitaService citaService;
    @Autowired private MascotaService mascotaService;

    @GetMapping
    public String listar(Model model) {
        model.addAttribute("listaCitas", citaService.obtenerTodas());
        return "citas/lista";
    }

    @GetMapping("/nueva")
    public String formulario(Model model) {
        model.addAttribute("cita", new Cita());
        model.addAttribute("listaMascotas", mascotaService.obtenerTodas());
        return "citas/formulario";
    }

    @PostMapping("/guardar")
    public String guardar(@ModelAttribute Cita cita) {
        citaService.guardar(cita);
        return "redirect:/citas";
    }

    @GetMapping("/{id}/editar")
    public String editar(@PathVariable Long id, Model model) {
        model.addAttribute("cita", citaService.buscarPorId(id));
        model.addAttribute("listaMascotas", mascotaService.obtenerTodas());
        return "citas/formulario";
    }

    @PostMapping("/{id}/eliminar")
    public String eliminar(@PathVariable Long id) {
        citaService.eliminar(id);
        return "redirect:/citas";
    }

    @PostMapping("/{id}/estado")
    public String cambiarEstado(@PathVariable Long id, @RequestParam String estado) {
        citaService.cambiarEstado(id, estado);
        return "redirect:/citas";
    }
}