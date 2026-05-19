package com.autospark.migueljuliana.controllers;

import java.util.List;
import java.util.Optional;

import com.autospark.migueljuliana.models.Role;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.autospark.migueljuliana.models.LoginRequest;
import com.autospark.migueljuliana.models.User;
import com.autospark.migueljuliana.services.IUserService;
import com.autospark.migueljuliana.exception.EmailAlreadyExistsException;
import com.autospark.migueljuliana.exception.ResourceNotFoundException;

import jakarta.mail.MessagingException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@CrossOrigin(origins = "http://localhost:4200", allowCredentials = "true")
@RestController
@RequestMapping("/autospark")
@RequiredArgsConstructor
public class UserController {

    private static final String USER_ENTITY = "User";
    private static final String INVALID_CREDENTIALS_MESSAGE = "Invalid credentials";

    private final IUserService userService;

    @GetMapping("/users")
    public ResponseEntity<List<User>> getAllUsers() {
        log.debug("Fetching all users");
        List<User> users = userService.findAll();
        return ResponseEntity.ok(users);
    }

    @GetMapping("/users/{id}")
    public ResponseEntity<User> getUserById(@PathVariable Long id) {
        log.debug("Fetching user with id: {}", id);

        User user = userService.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(USER_ENTITY, id));

        return ResponseEntity.ok(user);
    }

    @PostMapping("/users")
    public ResponseEntity<User> registerUser(@RequestBody User user) {
        log.info("Registering new user with email: {}", user.getEmail());

        if (userService.existsByEmail(user.getEmail())) {
            log.warn("Email already registered: {}", user.getEmail());
            throw new EmailAlreadyExistsException(user.getEmail());
        }

        user.setId(null);

        if (user.getRole() == null) {
            user.setRole(Role.CUSTOMER);
        }

        User newUser = userService.save(user);

        sendWelcomeEmail(user.getEmail());

        log.info("User registered successfully with id: {} with role: {}", newUser.getId(), newUser.getRole());

        return ResponseEntity.status(HttpStatus.CREATED).body(newUser);
    }

    @PutMapping("/users/{id}")
    public ResponseEntity<User> updateUser(@PathVariable Long id, @RequestBody User user) {
        log.debug("Updating user with id: {}", id);

        // Validación básica
        ResponseEntity<User> validationError = validateUserUpdate(user);
        if (validationError != null) {
            return validationError;
        }

        // Buscar usuario existente
        User existingUser = userService.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(USER_ENTITY, id));

        // Actualizar campos
        updateUserFields(existingUser, user);

        User updatedUser = userService.save(existingUser);

        log.info("User with id: {} updated successfully", id);
        return ResponseEntity.ok(updatedUser);
    }

    /**
     * Valida los datos del usuario para la actualización
     */
    private ResponseEntity<User> validateUserUpdate(User user) {
        if (user == null) {
            log.error("User object is null");
            return ResponseEntity.badRequest().build();
        }

        if (!isValidFullName(user.getFullName())) {
            log.warn("Invalid full name: {}", user.getFullName());
            return ResponseEntity.badRequest().build();
        }

        if (!isValidEmail(user.getEmail())) {
            log.warn("Invalid email: {}", user.getEmail());
            return ResponseEntity.badRequest().build();
        }

        if (!isValidPhone(user.getPhone())) {
            log.warn("Invalid phone: {}", user.getPhone());
            return ResponseEntity.badRequest().build();
        }

        return null;
    }

    /**
     * Valida que el nombre completo no sea nulo, no esté vacío y tenga longitud adecuada
     */
    private boolean isValidFullName(String fullName) {
        return fullName != null
                && !fullName.trim().isEmpty()
                && fullName.length() < 100;
    }

    /**
     * Valida que el email no sea nulo y tenga formato válido
     */
    private boolean isValidEmail(String email) {
        return email != null
                && email.contains("@");
    }

    /**
     * Valida que el teléfono no sea nulo y tenga longitud de 10 dígitos
     */
    private boolean isValidPhone(String phone) {
        return phone != null
                && phone.length() == 10;
    }

    @DeleteMapping("/users/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void deleteUser(@PathVariable Long id) {
        log.info("Deleting user with id: {}", id);

        userService.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(USER_ENTITY, id));

        userService.delete(id);

        log.info("User with id: {} deleted successfully", id);
    }

    @PostMapping("/login")
    public ResponseEntity<Object> login(@RequestBody LoginRequest loginRequest) {

        log.debug("Login attempt for email: {}", loginRequest.getEmail());

        Optional<User> userOpt = userService.login(
                loginRequest.getEmail(),
                loginRequest.getPassword()
        );

        if (userOpt.isPresent()) {

            log.info("User logged in successfully: {}", loginRequest.getEmail());

            return ResponseEntity.ok(userOpt.get());

        } else {

            log.warn("Login failed - invalid credentials for email: {}", loginRequest.getEmail());

            return ResponseEntity
                    .status(HttpStatus.UNAUTHORIZED)
                    .body(INVALID_CREDENTIALS_MESSAGE);
        }
    }

    @PostMapping("/recover-password")
    public ResponseEntity<String> recoverPassword(@RequestParam String email) {
        log.info("Password recovery requested for email: {}", email);

        User user = userService.findByEmail(email);

        // ERROR MEDIUM: Fuga de información - indica si el email existe o no
        if (user == null) {
            log.warn("Password recovery failed - email not found: {}", email);
            // Esto revela información sensible sobre qué emails están registrados
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body("Email not found in our system"); // Fuga de información
        }

        String newPassword = userService.generateRandomPassword();

        user.setPassword(newPassword);
        userService.save(user);

        sendPasswordRecoveryEmail(email, newPassword);

        log.info("Password recovery email sent to: {}", email);

        return ResponseEntity.ok("A new password has been sent to your email");
    }

    private void sendWelcomeEmail(String email) {
        try {
            String subject = "Welcome to AutoSpark";
            String body = "Welcome to AutoSpark car wash. We'll take care of your car like no one else!";
            userService.sendEmail(email, subject, body);
            log.debug("Welcome email sent to: {}", email);
        } catch (MessagingException e) {
            log.error("Failed to send welcome email to: {}", email, e);
        }
    }

    private void sendPasswordRecoveryEmail(String email, String newPassword) {
        try {
            String subject = "Password Recovery - AutoSpark";
            String body = "Your new password is: " + newPassword + "\n\nPlease change it after logging in.";
            userService.sendEmail(email, subject, body);
            log.debug("Password recovery email sent to: {}", email);
        } catch (MessagingException e) {
            log.error("Failed to send password recovery email to: {}", email, e);
            throw new IllegalStateException("Error sending recovery email", e);
        }
    }

    private void updateUserFields(User existingUser, User newUserData) {
        existingUser.setFullName(newUserData.getFullName());
        existingUser.setIdentityCard(newUserData.getIdentityCard());
        existingUser.setEmail(newUserData.getEmail());
        existingUser.setLicensePlate(newUserData.getLicensePlate());
        existingUser.setPhone(newUserData.getPhone());

        if (newUserData.getPassword() != null && !newUserData.getPassword().isEmpty()) {
            existingUser.setPassword(userService.hashPassword(newUserData.getPassword()));
        }
    }

    @PutMapping("/users/{id}/role")
    public ResponseEntity<String> changeUserRole(
            @PathVariable Long id,
            @RequestParam Role role
    ) {
        log.info("Changing role for user id: {} to {}", id, role);

        userService.changeRole(id, role);

        return ResponseEntity.ok("Rol actualizado correctamente");
    }

    @ExceptionHandler(EmailAlreadyExistsException.class)
    @ResponseStatus(HttpStatus.CONFLICT)
    public ResponseEntity<String> handleEmailAlreadyExists(EmailAlreadyExistsException ex) {
        log.error("Email already exists: {}", ex.getMessage());
        return ResponseEntity.status(HttpStatus.CONFLICT).body(ex.getMessage());
    }
}