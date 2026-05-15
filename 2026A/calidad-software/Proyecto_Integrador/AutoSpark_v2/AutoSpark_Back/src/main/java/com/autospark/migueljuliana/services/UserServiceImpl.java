package com.autospark.migueljuliana.services;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.HexFormat;
import java.util.List;
import java.util.Optional;

import com.autospark.migueljuliana.exception.ResourceNotFoundException;
import com.autospark.migueljuliana.models.Role;
import com.autospark.migueljuliana.models.User;
import com.autospark.migueljuliana.repositories.IUserRepository;

import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@RequiredArgsConstructor
public class UserServiceImpl implements IUserService {

    private static final SecureRandom RANDOM = new SecureRandom();

    private static final String ALLOWED_CHARS =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    private static final String USER_ENTITY = "User";

    private final IUserRepository userRepository;

    private final JavaMailSender mailSender;

    @Override
    @Transactional(readOnly = true)
    public List<User> findAll() {
        log.debug("Fetching all users");

        return (List<User>) userRepository.findAll();
    }

    @Override
    @Transactional(readOnly = true)
    public Optional<User> findById(Long id) {
        log.debug("Fetching user with id: {}", id);

        return userRepository.findById(id);
    }

    @Override
    @Transactional
    public void delete(Long id) {
        log.info("Deleting user with id: {}", id);

        userRepository.deleteById(id);

        log.info("User with id: {} deleted successfully", id);
    }

    @Override
    @Transactional
    public User save(User user) {
        log.debug("Saving user with email: {}", user.getEmail());

        user.setPassword(hashPassword(user.getPassword()));

        return userRepository.save(user);
    }

    @Override
    public String hashPassword(String password) {

        try {

            MessageDigest digest = MessageDigest.getInstance("SHA-256");

            byte[] hash = digest.digest(password.getBytes(StandardCharsets.UTF_8));

            String hashedPassword = HexFormat.of().formatHex(hash);

            log.debug("Password hashed successfully");

            return hashedPassword;

        } catch (NoSuchAlgorithmException e) {

            log.error("Error hashing password: {}", e.getMessage());

            throw new IllegalStateException("Error encrypting password", e);
        }
    }

    @Override
    @Transactional
    public void update(User user, Long id) {
        log.debug("Updating user with id: {}", id);

        User existingUser = userRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(USER_ENTITY, id));

        existingUser.setFullName(user.getFullName());
        existingUser.setIdentityCard(user.getIdentityCard());
        existingUser.setEmail(user.getEmail());
        existingUser.setPhone(user.getPhone());
        existingUser.setLicensePlate(user.getLicensePlate());
        existingUser.setRole(user.getRole());

        if (user.getPassword() != null && !user.getPassword().isEmpty()) {

            existingUser.setPassword(hashPassword(user.getPassword()));
        }

        userRepository.save(existingUser);

        log.info("User with id: {} updated successfully", id);
    }

    @Override
    @Transactional(readOnly = true)
    public boolean existsByEmail(String email) {
        log.debug("Checking if email exists: {}", email);

        return userRepository.existsByEmail(email);
    }

    @Override
    public Optional<User> login(String email, String password) {
        log.debug("Login attempt for email: {}", email);

        Optional<User> userOpt = userRepository.findByEmail(email);

        if (userOpt.isPresent()) {

            String hashedPassword = hashPassword(password);

            if (userOpt.get().getPassword().equals(hashedPassword)) {

                log.info("Login successful for email: {}", email);

                return userOpt;
            }

            log.warn("Login failed - invalid password for email: {}", email);

        } else {

            log.warn("Login failed - email not found: {}", email);
        }

        return Optional.empty();
    }

    @Override
    public User findByEmail(String email) {
        log.debug("Finding user by email: {}", email);

        return userRepository.findByEmail(email).orElse(null);
    }

    @Override
    public void sendEmail(String to, String subject, String body) throws MessagingException {
        log.debug("Sending email to: {}", to);

        MimeMessage message = mailSender.createMimeMessage();

        MimeMessageHelper helper = new MimeMessageHelper(message, true);

        helper.setTo(to);
        helper.setSubject(subject);
        helper.setText(body, true);
        helper.setFrom("servicioalclienteautospark@gmail.com");

        mailSender.send(message);

        log.info("Email sent successfully to: {}", to);
    }

    @Override
    public String generateRandomPassword() {

        StringBuilder password = new StringBuilder(8);

        for (int i = 0; i < 8; i++) {

            int index = RANDOM.nextInt(ALLOWED_CHARS.length());

            password.append(ALLOWED_CHARS.charAt(index));
        }

        log.debug("Random password generated");

        return password.toString();
    }

    @Override
    @Transactional
    public void changeRole(Long id, Role role) {

        User existingUser = userRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(USER_ENTITY, id));

        existingUser.setRole(role);

        userRepository.save(existingUser);

        log.info("Rol actualizado correctamente para usuario id: {}", id);
    }
}