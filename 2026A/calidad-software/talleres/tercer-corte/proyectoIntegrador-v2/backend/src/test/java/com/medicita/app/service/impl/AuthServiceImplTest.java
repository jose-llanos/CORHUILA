package com.medicita.app.service.impl;

import com.medicita.app.dto.auth.AuthResponse;
import com.medicita.app.dto.auth.LoginRequest;
import com.medicita.app.dto.auth.RegisterRequest;
import com.medicita.app.entity.Patient;
import com.medicita.app.entity.User;
import com.medicita.app.enums.Role;
import com.medicita.app.repository.PatientRepository;
import com.medicita.app.repository.UserRepository;
import com.medicita.app.security.JwtTokenProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.time.LocalDate;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/*
 * Pruebas unitarias para AuthServiceImpl.
 *
 * Acá probamos el registro y el login de usuarios sin tocar la base de datos real.
 * Para eso usamos Mockito: en vez de inyectar el UserRepository de verdad, le metemos
 * un "doble" (mock) al que le decimos cómo responder en cada caso.
 *
 * La idea es cubrir los 3 tipos de escenarios que pide el PDF:
 *   - Casos exitosos (todo sale bien).
 *   - Casos límite (situaciones raras pero válidas, como cuenta desactivada).
 *   - Casos de error (cuando el servicio debe lanzar excepciones).
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("AuthServiceImpl — Pruebas unitarias")
class AuthServiceImplTest {

    // Estas son las dependencias del AuthServiceImpl. Las "fingimos" con @Mock
    // para que no toquen BD ni hagan cifrado real ni generen JWT real.
    @Mock
    private UserRepository userRepository;

    @Mock
    private PatientRepository patientRepository;

    @Mock
    private PasswordEncoder passwordEncoder;

    @Mock
    private JwtTokenProvider jwtTokenProvider;

    // Mockito inyecta los mocks de arriba dentro del AuthServiceImpl real
    // que vamos a probar.
    @InjectMocks
    private AuthServiceImpl authService;

    // Objetos de prueba que reutilizamos en varios tests. Se reinician
    // antes de cada uno gracias al @BeforeEach.
    private RegisterRequest validRegisterRequest;
    private LoginRequest validLoginRequest;
    private User existingUser;

    /*
     * Se ejecuta antes de cada @Test. Aquí dejamos los datos "limpios" para que
     * un test no afecte a otro (cada prueba debe ser independiente).
     */
    @BeforeEach
    void setUp() {
        // Petición de registro con todos los campos válidos.
        validRegisterRequest = RegisterRequest.builder()
                .firstName("Juan")
                .lastName("Pérez")
                .email("juan.perez@medicita.com")
                .password("SuperSecreto2026*")
                .documentNumber("1075123456")
                .phone("3001234567")
                .birthDate(LocalDate.of(1995, 5, 15))
                .build();

        // Petición de login con las mismas credenciales del registro.
        validLoginRequest = new LoginRequest();
        validLoginRequest.setEmail("juan.perez@medicita.com");
        validLoginRequest.setPassword("SuperSecreto2026*");

        // Usuario que finge estar "ya guardado" en la BD para los tests de login.
        // El password viene en formato cifrado (hash de BCrypt) porque así
        // estaría realmente almacenado en producción.
        existingUser = User.builder()
                .id(UUID.randomUUID())
                .firstName("Juan")
                .lastName("Pérez")
                .email("juan.perez@medicita.com")
                .password("$2a$10$hashedpassword")
                .role(Role.PATIENT)
                .active(true)
                .build();
    }

    // =========================================================================
    // Pruebas del método register()
    // =========================================================================

    /*
     * Caso feliz del registro: si el email es nuevo, el servicio debe guardar
     * el usuario, generar un JWT y devolver el AuthResponse completo.
     */
    @Test
    @DisplayName("register: registra usuario nuevo y devuelve AuthResponse con token")
    void register_conDatosValidos_devuelveAuthResponseConToken() {
        // Configuramos los mocks: email NO existe, password se "cifra" como una cadena
        // fija, y el JWT que generamos siempre es "jwt-token-xyz".
        when(userRepository.existsByEmail(validRegisterRequest.getEmail())).thenReturn(false);
        when(passwordEncoder.encode(anyString())).thenReturn("$2a$10$hashedpassword");
        when(jwtTokenProvider.generateToken(any(User.class))).thenReturn("jwt-token-xyz");

        // Ejecutamos el método que queremos probar.
        AuthResponse response = authService.register(validRegisterRequest);

        // Verificamos que la respuesta venga con todo lo que esperamos.
        assertThat(response).isNotNull();
        assertThat(response.getToken()).isEqualTo("jwt-token-xyz");
        assertThat(response.getEmail()).isEqualTo("juan.perez@medicita.com");
        assertThat(response.getRole()).isEqualTo("PATIENT");
        assertThat(response.getFullName()).isEqualTo("Juan Pérez");
    }

    /*
     * Prueba de seguridad: nos aseguramos de que el password NO se guarde
     * tal cual como lo escribió el usuario. Si esto fallara, sería un
     * problema gravísimo de seguridad.
     */
    @Test
    @DisplayName("register: cifra el password antes de persistirlo")
    void register_cifraElPasswordAntesDePersistir() {
        when(userRepository.existsByEmail(anyString())).thenReturn(false);
        when(passwordEncoder.encode("SuperSecreto2026*")).thenReturn("$2a$10$hashedpassword");
        when(jwtTokenProvider.generateToken(any(User.class))).thenReturn("token");

        authService.register(validRegisterRequest);

        // El ArgumentCaptor "atrapa" el User que el servicio le pasó al save()
        // para que podamos revisarlo después.
        ArgumentCaptor<User> userCaptor = ArgumentCaptor.forClass(User.class);
        verify(userRepository).save(userCaptor.capture());
        User savedUser = userCaptor.getValue();

        // El password guardado debe ser el hash, NUNCA el texto plano.
        assertThat(savedUser.getPassword()).isEqualTo("$2a$10$hashedpassword");
        assertThat(savedUser.getPassword()).isNotEqualTo("SuperSecreto2026*");
    }

    /*
     * Regla de negocio: todo usuario que se registra por el endpoint público
     * es un PACIENTE. Doctores y admins se crean por otro flujo, así que
     * acá no debería poder colarse un rol distinto.
     */
    @Test
    @DisplayName("register: asigna rol PATIENT por defecto al nuevo usuario")
    void register_asignaRolPatientPorDefecto() {
        when(userRepository.existsByEmail(anyString())).thenReturn(false);
        when(passwordEncoder.encode(anyString())).thenReturn("hashed");
        when(jwtTokenProvider.generateToken(any(User.class))).thenReturn("token");

        authService.register(validRegisterRequest);

        ArgumentCaptor<User> userCaptor = ArgumentCaptor.forClass(User.class);
        verify(userRepository).save(userCaptor.capture());

        assertThat(userCaptor.getValue().getRole()).isEqualTo(Role.PATIENT);
    }

    /*
     * El registro debe crear DOS cosas: el User (credenciales) y el Patient
     * (datos médicos). Acá nos aseguramos de que el Patient también se guarde
     * y de que los datos pasen completos al repositorio.
     */
    @Test
    @DisplayName("register: persiste también la entidad Patient asociada al usuario")
    void register_persisteEntidadPatient() {
        when(userRepository.existsByEmail(anyString())).thenReturn(false);
        when(passwordEncoder.encode(anyString())).thenReturn("hashed");
        when(jwtTokenProvider.generateToken(any(User.class))).thenReturn("token");

        authService.register(validRegisterRequest);

        ArgumentCaptor<Patient> patientCaptor = ArgumentCaptor.forClass(Patient.class);
        verify(patientRepository).save(patientCaptor.capture());
        Patient savedPatient = patientCaptor.getValue();

        // Revisamos que los datos del paciente coincidan con lo que llegó en el request.
        assertThat(savedPatient.getDocumentNumber()).isEqualTo("1075123456");
        assertThat(savedPatient.getPhone()).isEqualTo("3001234567");
        assertThat(savedPatient.getBirthDate()).isEqualTo(LocalDate.of(1995, 5, 15));
        assertThat(savedPatient.getUser()).isNotNull();
    }

    /*
     * Caso de error: si alguien intenta registrarse con un email que ya existe,
     * el servicio debe lanzar excepción y NO guardar nada (ni el User ni el Patient).
     */
    @Test
    @DisplayName("register: lanza RuntimeException cuando el email ya está registrado")
    void register_conEmailDuplicado_lanzaRuntimeException() {
        // Simulamos que el email ya está ocupado.
        when(userRepository.existsByEmail(validRegisterRequest.getEmail())).thenReturn(true);

        // Esperamos que estalle con el mensaje correcto.
        assertThatThrownBy(() -> authService.register(validRegisterRequest))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("Email already registered")
                .hasMessageContaining("juan.perez@medicita.com");

        // Verificamos que NO se haya llamado save() en ninguno de los dos repos.
        // Esto es importante: queremos asegurar que no quedó nada a medias en BD.
        verify(userRepository, never()).save(any(User.class));
        verify(patientRepository, never()).save(any(Patient.class));
    }

    // =========================================================================
    // Pruebas del método login()
    // =========================================================================

    /*
     * Caso feliz del login: email correcto + password correcto + cuenta activa
     * → devuelve AuthResponse con el JWT generado.
     */
    @Test
    @DisplayName("login: con credenciales válidas devuelve AuthResponse con token")
    void login_conCredencialesValidas_devuelveAuthResponse() {
        when(userRepository.findByEmail("juan.perez@medicita.com"))
                .thenReturn(Optional.of(existingUser));
        // matches() compara el password plano contra el hash, no contra otro plano.
        when(passwordEncoder.matches("SuperSecreto2026*", "$2a$10$hashedpassword"))
                .thenReturn(true);
        when(jwtTokenProvider.generateToken(existingUser)).thenReturn("jwt-token-xyz");

        AuthResponse response = authService.login(validLoginRequest);

        assertThat(response).isNotNull();
        assertThat(response.getToken()).isEqualTo("jwt-token-xyz");
        assertThat(response.getEmail()).isEqualTo("juan.perez@medicita.com");
        assertThat(response.getRole()).isEqualTo("PATIENT");
        assertThat(response.getFullName()).isEqualTo("Juan Pérez");
    }

    /*
     * Caso de error: el email no está registrado. Importante: el mensaje
     * de error debe ser genérico ("Invalid email or password") para no
     * darle pistas a un atacante sobre qué emails existen en el sistema.
     */
    @Test
    @DisplayName("login: con email inexistente lanza BadCredentialsException")
    void login_conEmailInexistente_lanzaBadCredentialsException() {
        // findByEmail devuelve Optional vacío = email no encontrado.
        when(userRepository.findByEmail(anyString())).thenReturn(Optional.empty());

        assertThatThrownBy(() -> authService.login(validLoginRequest))
                .isInstanceOf(BadCredentialsException.class)
                .hasMessage("Invalid email or password");
    }

    /*
     * Caso de error: el email sí existe pero el password no coincide.
     * Mismo mensaje genérico que el caso anterior (por seguridad).
     */
    @Test
    @DisplayName("login: con password incorrecto lanza BadCredentialsException")
    void login_conPasswordIncorrecto_lanzaBadCredentialsException() {
        when(userRepository.findByEmail(anyString())).thenReturn(Optional.of(existingUser));
        // El password no coincide con el hash guardado.
        when(passwordEncoder.matches(anyString(), anyString())).thenReturn(false);

        assertThatThrownBy(() -> authService.login(validLoginRequest))
                .isInstanceOf(BadCredentialsException.class)
                .hasMessage("Invalid email or password");
    }

    /*
     * Caso límite: las credenciales son correctas PERO la cuenta está
     * desactivada (un admin la deshabilitó). Aquí sí cambia el mensaje
     * porque al usuario sí le interesa saber que su cuenta fue bloqueada.
     */
    @Test
    @DisplayName("login: con cuenta desactivada lanza BadCredentialsException")
    void login_conCuentaDesactivada_lanzaBadCredentialsException() {
        existingUser.setActive(false);
        when(userRepository.findByEmail(anyString())).thenReturn(Optional.of(existingUser));
        when(passwordEncoder.matches(anyString(), anyString())).thenReturn(true);

        assertThatThrownBy(() -> authService.login(validLoginRequest))
                .isInstanceOf(BadCredentialsException.class)
                .hasMessage("Account is disabled");
    }
}
