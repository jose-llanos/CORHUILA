package com.corhuila.gestionpruebas.controller;

import com.corhuila.gestionpruebas.model.Mascota;
import com.corhuila.gestionpruebas.service.DuenioService;
import com.corhuila.gestionpruebas.service.MascotaService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping("/mascotas")
public class MascotaController {

    @Autowired private MascotaService mascotaService;
    @Autowired private DuenioService duenioService;

    @GetMapping
    public String listar(Model model) {
        model.addAttribute("listaMascotas", mascotaService.obtenerTodas());
        return "mascotas/lista";
    }

    @GetMapping("/nueva")
    public String formulario(Model model) {
        model.addAttribute("mascota", new Mascota());
        model.addAttribute("listaDuenios", duenioService.obtenerTodos());
        return "mascotas/formulario";
    }

    @PostMapping("/guardar")
    public String guardar(@ModelAttribute Mascota mascota) {
        mascotaService.guardar(mascota);
        return "redirect:/mascotas";
    }

    @GetMapping("/{id}/editar")
    public String editar(@PathVariable Long id, Model model) {
        model.addAttribute("mascota", mascotaService.buscarPorId(id));
        model.addAttribute("listaDuenios", duenioService.obtenerTodos());
        return "mascotas/formulario";
    }

    @PostMapping("/{id}/eliminar")
    public String eliminar(@PathVariable Long id) {
        mascotaService.eliminar(id);
        return "redirect:/mascotas";
    }
}