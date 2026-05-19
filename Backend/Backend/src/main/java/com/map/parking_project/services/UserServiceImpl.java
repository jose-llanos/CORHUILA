package com.map.parking_project.services;

import com.map.parking_project.models.User;
import com.map.parking_project.repositories.IUserRepository;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.security.SecureRandom;
import java.util.List;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import org.springframework.web.server.ResponseStatusException;

@Service
public class UserServiceImpl implements IUserService {

    @Autowired
    private IUserRepository userDao; // Repositorio para manejar la entidad User

    @Autowired
    private JavaMailSender mailSender; // Servicio para enviar correos electrónicos

    @Override
    @Transactional
    public List<User> findAll() {
        List<User> users = new ArrayList<>();
        userDao.findAll().forEach(users::add);
        return users; // Obtiene todos los usuarios de la base de datos
    }

    @Override
    @Transactional
    public User findById(Long id) {
        return userDao.findById(id).orElse(null); // Busca un usuario por su ID, si no lo encuentra devuelve null
    }

    @Override
    @Transactional
    public User save(User user) {
        return userDao.save(user); // Guarda o actualiza un usuario en la base de datos
    }

    @Override
    public User saveUser(User user) {
        user.setPassword(ContraseniaSha256(user.getPassword())); // Encripta la contraseña antes de guardar
        return userDao.save(user);
    }

    @Override
    public String ContraseniaSha256(String password) {
        try {
            MessageDigest instancia = MessageDigest.getInstance("SHA-256"); // Crea una instancia de SHA-256
            byte[] hash = instancia.digest(password.getBytes(StandardCharsets.UTF_8)); // Genera el hash de la contraseña
            return HexFormat.of().formatHex(hash); // Convierte el hash en una cadena hexadecimal
        } catch (NoSuchAlgorithmException e) {
            // ✅ CORREGIDO: Se cambia RuntimeException por IllegalStateException y se pasa la excepción original 'e'
            throw new IllegalStateException("Error al encriptar la contraseña: el algoritmo SHA-256 no está disponible", e);
        }
    }

    @Override
    @Transactional
    public void delete(Long id) {
        userDao.deleteById(id); // Elimina un usuario por su ID
    }

    @Override
    @org.springframework.transaction.annotation.Transactional(readOnly = true)
    public boolean existsByEmail(String email) {
        return userDao.existsByEmail(email); // Verifica si un correo ya está registrado en la base de datos
    }

    @Override
    public User findByEmail(String email) {
        return userDao.findByEmail(email).orElse(null); // Busca un usuario por su correo electrónico
    }

    public void sendEmail(String to, String subject, String body) throws MessagingException {
        MimeMessage message = mailSender.createMimeMessage();
        MimeMessageHelper helper = new MimeMessageHelper(message, true);

        helper.setTo(to); // Establece el destinatario del correo
        helper.setSubject(subject); // Establece el asunto del correo
        helper.setText(body, true); // true = Permite contenido HTML en el cuerpo del correo
        helper.setFrom("MAP_parking@gmail.com"); // Remitente del correo

        mailSender.send(message); // Envía el correo
    }

    public String generarContraseniaAleatoria() {
        String caracteres = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"; // Conjunto de caracteres permitidos
        SecureRandom random = new SecureRandom();
        StringBuilder contraseña = new StringBuilder(8); // Genera una contraseña de 8 caracteres

        for (int i = 0; i < 8; i++) {
            int indice = random.nextInt(caracteres.length());
            // Selecciona un carácter aleatorio
            contraseña.append(caracteres.charAt(indice));
        }
        return contraseña.toString();
        // Retorna la contraseña generada
    }

    public User findByPlate(String plate) {
        return userDao.findByPlate(plate)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Usuario no encontrado con la placa: " + plate));
    }
}