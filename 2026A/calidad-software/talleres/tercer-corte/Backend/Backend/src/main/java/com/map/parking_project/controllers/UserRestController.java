package com.map.parking_project.controllers;

import com.map.parking_project.models.User;
import com.map.parking_project.services.IUserService;
import jakarta.mail.MessagingException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import com.map.parking_project.dto.UserDTO;
import com.map.parking_project.dto.ValidarTarifaDTO;

import java.util.List;
import java.util.Map;

@CrossOrigin(origins = {"http://localhost:4200"})
@RestController
@RequestMapping("/api")
public class UserRestController {

    @Autowired
    private IUserService userService;

    // 🔹 CONSTANTES PARA ESTANDARIZAR JSON Y EVITAR CODE SMELLS
    private static final String MESSAGE_KEY = "message";
    private static final String ERROR_KEY = "error";

    @GetMapping("/user")
    public List<User> index() {
        return userService.findAll();
    }

    @GetMapping("/user/{id}")
    public ResponseEntity<User> show(@PathVariable Long id) {
        User user = userService.findById(id);
        return user != null ? ResponseEntity.ok(user) : ResponseEntity.notFound().build();
    }

    @PostMapping("/user")
    @ResponseStatus(HttpStatus.CREATED)
    public User create(@RequestBody UserDTO userDTO) {
        User newUser = new User();
        newUser.setName(userDTO.getName());
        newUser.setLastname(userDTO.getLastname());
        newUser.setPhone(userDTO.getPhone());
        newUser.setPlate(userDTO.getPlate());
        newUser.setTypecar(userDTO.getTypecar());
        newUser.setEmail(userDTO.getEmail());
        newUser.setPassword(userDTO.getPassword());
        newUser.setRol(userDTO.getRol());

        return userService.saveUser(newUser);
    }

    @PutMapping("/user/{id}")
    public ResponseEntity<User> update(@RequestBody UserDTO userDTO, @PathVariable Long id) {
        User currentUser = userService.findById(id);

        if (currentUser == null) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).build();
        }

        currentUser.setName(userDTO.getName());
        currentUser.setLastname(userDTO.getLastname());
        currentUser.setPhone(userDTO.getPhone());
        currentUser.setPlate(userDTO.getPlate());
        currentUser.setTypecar(userDTO.getTypecar());
        currentUser.setEmail(userDTO.getEmail());
        currentUser.setPassword(userDTO.getPassword());
        currentUser.setRol(userDTO.getRol());

        return ResponseEntity.status(HttpStatus.CREATED).body(userService.save(currentUser));
    }

    @DeleteMapping("/user/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void delete(@PathVariable Long id) {
        userService.delete(id);
    }

    @PostMapping("/recuperarcontrasenia")
    public ResponseEntity<Map<String, String>> recuperarContrasenia(@RequestParam String email) {
        User user = userService.findByEmail(email);

        if (user == null) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(Map.of(ERROR_KEY, "Correo no registrado"));
        }

        String nuevaContrasenia = userService.generarContraseniaAleatoria();
        user.setPassword(userService.ContraseniaSha256(nuevaContrasenia));
        userService.save(user);

        String asunto = "Recuperación de contraseña";
        String cuerpo = "Tu nueva contraseña es: " + nuevaContrasenia;
        try {
            userService.sendEmail(email, asunto, cuerpo);
            return ResponseEntity.ok(Map.of(MESSAGE_KEY, "Se ha enviado un correo con la nueva contraseña."));
        } catch (MessagingException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(Map.of(ERROR_KEY, "Error al enviar el correo."));
        }
    }

    @GetMapping("/send")
    public ResponseEntity<Map<String, String>> sendEmail(
            @RequestParam String to,
            @RequestParam String subject,
            @RequestParam String body) {
        try {
            userService.sendEmail(to, subject, body);
            return ResponseEntity.ok(Map.of(MESSAGE_KEY, "Correo enviado correctamente a " + to));
        } catch (MessagingException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(Map.of(ERROR_KEY, "Error enviando el correo: " + e.getMessage()));
        }
    }

    @PostMapping("/validar-tarifa")
    public ResponseEntity<Map<String, Object>> validarYCalcularTarifa(@RequestBody ValidarTarifaDTO request) {
        try {
            User user = userService.findByPlate(request.getPlate());

            if (user == null) {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body(Map.of(ERROR_KEY, "Vehículo no encontrado."));
            }

            if (!user.getTypecar().equals(request.getTypecar())) {
                // ✅ CORRECCIÓN 1: Reemplazado RuntimeException por IllegalArgumentException
                throw new IllegalArgumentException("El tipo de vehículo no coincide con el registrado.");
            }

            double tarifaBase;
            switch (user.getTypecar()) {
                case "Automóvil":
                    tarifaBase = 3.5;
                    break;
                case "Camioneta":
                    tarifaBase = 5.0;
                    break;
                case "Moto":
                    tarifaBase = 2.5;
                    break;
                default:
                    // ✅ CORRECCIÓN 2: Reemplazado RuntimeException por IllegalArgumentException
                    throw new IllegalArgumentException("Tipo de vehículo no válido.");
            }

            double total = tarifaBase * request.getHours();

            return ResponseEntity.ok(Map.of("total", total, MESSAGE_KEY, "Tarifa calculada con éxito"));

        // ✅ CORRECCIÓN 3: El catch ahora captura específicamente IllegalArgumentException
        } catch (IllegalArgumentException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(Map.of(ERROR_KEY, e.getMessage()));
        }
    }
}