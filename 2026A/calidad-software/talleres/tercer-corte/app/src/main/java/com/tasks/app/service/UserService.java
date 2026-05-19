package com.tasks.app.service;

import com.tasks.app.dto.request.LoginRequest;
import com.tasks.app.dto.request.RegisterRequest;
import com.tasks.app.dto.response.AuthResponse;
import com.tasks.app.dto.response.UserProfileResponse;
import com.tasks.app.entity.User;
import com.tasks.app.exception.ConflictException;
import com.tasks.app.exception.UnauthorizedException;
import com.tasks.app.repository.UserRepository;
import com.tasks.app.security.JwtService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;

    @Transactional
    public UserProfileResponse register(RegisterRequest request) {
        if (userRepository.existsByUsername(request.getUsername())) {
            throw new ConflictException("El username ya está en uso");
        }
        if (userRepository.existsByEmail(request.getEmail())) {
            throw new ConflictException("El email ya está en uso");
        }
        User user = User.builder()
                .username(request.getUsername())
                .email(request.getEmail())
                .password(passwordEncoder.encode(request.getPassword()))
                .build();
        return UserProfileResponse.from(userRepository.save(user));
    }

    @Transactional(readOnly = true)
    public AuthResponse login(LoginRequest request) {
        User user = userRepository.findByUsername(request.getUsername())
                .orElseThrow(() -> new UnauthorizedException("Credenciales inválidas"));
        if (!passwordEncoder.matches(request.getPassword(), user.getPassword())) {
            throw new UnauthorizedException("Credenciales inválidas");
        }
        return new AuthResponse(jwtService.generateToken(user));
    }

    @Transactional(readOnly = true)
    public UserProfileResponse getProfile(User currentUser) {
        return UserProfileResponse.from(currentUser);
    }
}
