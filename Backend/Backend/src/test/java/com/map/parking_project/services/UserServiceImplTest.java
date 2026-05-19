package com.map.parking_project.services;
import com.map.parking_project.models.User;
import com.map.parking_project.repositories.IUserRepository;
import jakarta.mail.internet.MimeMessage;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.mail.javamail.JavaMailSender;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;
@ExtendWith(MockitoExtension.class)

public class UserServiceImplTest {
    @Mock
    private IUserRepository userRepository;
    @Mock
    private JavaMailSender mailSender;
    @InjectMocks
    private UserServiceImpl userService;
    private User user;
    @BeforeEach
    void setUp() {
        user = new User();
        user.setId(1L);
        user.setEmail("test@test.com");
        user.setPassword("12345");
    }
    @Test
    void testFindAll() {
        when(userRepository.findAll()).thenReturn(List.of(user));
        List<User> result = userService.findAll();
        assertNotNull(result);
        assertEquals(1, result.size());
    }
    @Test
    void testSaveUser_EncryptsPassword() {
        when(userRepository.save(any(User.class))).thenReturn(user);
        User savedUser = userService.saveUser(user);
        // Verifica que la contraseña guardada no sea la original (está en SHA-256)
        assertNotEquals("12345", savedUser.getPassword());
        verify(userRepository, times(1)).save(any(User.class));
    }
    @Test
    void testContraseniaSha256() {
        String pass = "hola123";
        String hash = userService.ContraseniaSha256(pass);
        assertNotNull(hash);
        assertEquals(64, hash.length()); // SHA-256 siempre tiene 64 caracteres hex
    }
    @Test
    void testGenerarContraseniaAleatoria() {
        String pass1 = userService.generarContraseniaAleatoria();
        String pass2 = userService.generarContraseniaAleatoria();
        assertNotNull(pass1);
        assertEquals(8, pass1.length());
        assertNotEquals(pass1, pass2); // Deben ser diferentes
    }
    @Test
    void testSendEmail() throws Exception {
        MimeMessage mimeMessage = mock(MimeMessage.class);
        when(mailSender.createMimeMessage()).thenReturn(mimeMessage);
        userService.sendEmail("to@test.com", "Asunto", "Cuerpo");
        verify(mailSender, times(1)).send(any(MimeMessage.class));
    }
}
