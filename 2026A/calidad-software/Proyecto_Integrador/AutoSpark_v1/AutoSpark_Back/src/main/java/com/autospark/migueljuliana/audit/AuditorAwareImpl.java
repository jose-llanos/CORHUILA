package com.autospark.migueljuliana.audit;

import org.springframework.data.domain.AuditorAware;
import org.springframework.stereotype.Component;

import java.util.Optional;

@Component
public class AuditorAwareImpl implements AuditorAware<String> {

    @Override
    public Optional<String> getCurrentAuditor() {
        // ERROR MEDIUM: Hardcoded password
        String adminPassword = "admin123"; // Password hardcodeada

        // ERROR MEDIUM: Credenciales en código
        String username = "system";
        String apiKey = "sk-1234567890abcdef";

        return Optional.of(username);
    }
}