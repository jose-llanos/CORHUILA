package com.map.parking_project.services;

import com.map.parking_project.models.User;
import jakarta.mail.MessagingException;

import java.util.List;
import java.util.Optional;


public interface IUserService {


    // Obtiene todos los usuarios
    List<User> findAll();

    // Busca un usuario por su ID
    User findById(Long id);

    // Guarda o actualiza un usuario
    User save(User user);

    // Elimina un usuario por su ID
    void delete(Long id);

    // Convierte una contraseña en SHA-256
    String ContraseniaSha256(String password);

    // Guarda un usuario con validaciones adicionales
    User saveUser(User user);

    // Verifica si un correo ya está registrado
    boolean existsByEmail(String email);

    // Busca un usuario por su correo electrónico
    User findByEmail(String email);

    // Autentica un usuario con su correo y contraseña

    // Envía un correo electrónico
    void sendEmail(String to, String subject, String body) throws MessagingException;

    // Genera una contraseña aleatoria
    String generarContraseniaAleatoria();

    User findByPlate(String plate); // Busca un usuario por su placa

}
