package com.autospark.migueljuliana.services;

import java.util.List;
import java.util.Optional;

import com.autospark.migueljuliana.models.Role;
import com.autospark.migueljuliana.models.User;

import jakarta.mail.MessagingException;

public interface IUserService {

    List<User> findAll();

    Optional<User> findById(Long id);

    User save(User user);

    void update(User user, Long id);

    void delete(Long id);

    String hashPassword(String password);

    boolean existsByEmail(String email);

    User findByEmail(String email);

    Optional<User> login(String email, String password);

    void sendEmail(String to, String subject, String body) throws MessagingException;

    String generateRandomPassword();

    void changeRole(Long id, Role role);
}