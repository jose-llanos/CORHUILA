package com.corhuila.gestionpruebas.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class HomeController {

    @GetMapping("/")
    public String index() {
        return "index";
    }

    // ✅ Método para mostrar el formulario de login
    @GetMapping("/login")
    public String login() {
        return "login";
    }
}