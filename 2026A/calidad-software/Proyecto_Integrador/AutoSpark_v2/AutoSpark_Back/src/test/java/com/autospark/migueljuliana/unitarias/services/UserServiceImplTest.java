package com.autospark.migueljuliana.unitarias.services;

import com.autospark.migueljuliana.models.User;
import com.autospark.migueljuliana.models.Role;
import com.autospark.migueljuliana.repositories.IUserRepository;
import com.autospark.migueljuliana.services.UserServiceImpl;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * Suite de pruebas unitarias para el servicio de gestión de usuarios.
 *
 * <p>Esta clase prueba los siguientes requisitos funcionales:</p>
 * <ul>
 *   <li>RF1: Registro de usuarios - TC-001 a TC-003</li>
 *   <li>RF2: Gestión completa de usuarios (CRUD) - TC-004 a TC-010</li>
 *   <li>RF3: Recuperar contraseña por email - TC-011 a TC-013</li>
 *   <li>RF4: Administrador puede promover usuarios - TC-014</li>
 * </ul>
 *
 * @author AutoSpark Team
 * @version 1.0
 */
@ExtendWith(MockitoExtension.class)
class UserServiceImplTest {

    // ===== MOCKS =====
    @Mock
    private IUserRepository userRepository;

    @InjectMocks
    private UserServiceImpl userService;

    // ===== OBJETOS DE PRUEBA =====
    private User testUser;

    // ===== CONFIGURACIÓN INICIAL =====
    @BeforeEach
    void setUp() {
        testUser = new User();
        testUser.setId(1L);
        testUser.setFullName("Juan Perez");
        testUser.setIdentityCard("12345678");
        testUser.setEmail("juan@test.com");
        testUser.setPhone("3001234567");
        testUser.setPassword("Password123");
        testUser.setLicensePlate("ABC123");
        testUser.setRole(Role.CUSTOMER);
    }

    // ================================================================
    // ========== PRUEBAS PARA RF1: REGISTRO DE USUARIOS ==========
    // ================================================================

    /**
     * TC-001: Registro exitoso
     * <p><b>Requisito:</b> RF1 - El sistema debe permitir registrar un nuevo usuario</p>
     * <p><b>Entrada:</b> Objeto User con datos válidos (email no registrado)</p>
     * <p><b>Salida esperada:</b> Usuario guardado con sus datos correctos</p>
     */
    @Test
    void testRegistroUsuarioExitoso() {
        when(userRepository.save(any(User.class))).thenReturn(testUser);

        User savedUser = userService.save(testUser);

        assertNotNull(savedUser);
        assertEquals("juan@test.com", savedUser.getEmail());
        assertEquals(Role.CUSTOMER, savedUser.getRole());
        verify(userRepository, times(1)).save(any(User.class));
    }

    /**
     * TC-002: Validar email duplicado
     * <p><b>Requisito:</b> RF1 - El sistema debe validar que el email no esté duplicado</p>
     * <p><b>Entrada:</b> Email que ya existe en la base de datos</p>
     * <p><b>Salida esperada:</b> true (el email ya está registrado)</p>
     */
    @Test
    void testExistsByEmail() {
        when(userRepository.existsByEmail("juan@test.com")).thenReturn(true);

        boolean exists = userService.existsByEmail("juan@test.com");

        assertTrue(exists);
        verify(userRepository, times(1)).existsByEmail("juan@test.com");
    }

    /**
     * TC-003: Verificar encriptación SHA-256
     * <p><b>Requisito:</b> RF1 - El sistema debe encriptar contraseñas</p>
     * <p><b>Entrada:</b> Contraseña en texto plano "Password123"</p>
     * <p><b>Salida esperada:</b> Hash de 64 caracteres hexadecimales</p>
     */
    @Test
    void testHashPassword() {
        String password = "Password123";
        String hash = userService.hashPassword(password);
        assertNotNull(hash);
        assertNotEquals(password, hash);
        assertEquals(64, hash.length());
    }

    // ================================================================
    // ========== PRUEBAS PARA RF2: GESTIÓN COMPLETA DE USUARIOS ==========
    // ================================================================

    /**
     * TC-004: Listar todos los usuarios
     * <p><b>Requisito:</b> RF2 - El sistema debe permitir listar todos los usuarios</p>
     * <p><b>Entrada:</b> Ninguna</p>
     * <p><b>Salida esperada:</b> Lista de usuarios registrados</p>
     */
    @Test
    void testFindAllUsers() {
        when(userRepository.findAll()).thenReturn(java.util.List.of(testUser));

        var users = userService.findAll();

        assertNotNull(users);
        assertEquals(1, users.size());
        verify(userRepository, times(1)).findAll();
    }

    /**
     * TC-005: Eliminar usuario existente
     * <p><b>Requisito:</b> RF2 - El sistema debe permitir eliminar un usuario</p>
     * <p><b>Entrada:</b> ID = 1 (usuario existente)</p>
     * <p><b>Salida esperada:</b> Usuario eliminado del sistema</p>
     */
    @Test
    void testDeleteUserById() {
        doNothing().when(userRepository).deleteById(1L);
        userService.delete(1L);
        verify(userRepository, times(1)).deleteById(1L);
    }

    /**
     * TC-006: Login con email no registrado
     * <p><b>Requisito:</b> RF2 - Manejo de login con email no registrado</p>
     * <p><b>Entrada:</b> Email no registrado en el sistema</p>
     * <p><b>Salida esperada:</b> Optional vacío</p>
     */
    @Test
    void testLoginWithEmailNotFound() {
        when(userRepository.findByEmail("noexiste@test.com")).thenReturn(Optional.empty());

        Optional<User> result = userService.login("noexiste@test.com", "Password123");

        assertTrue(result.isEmpty());
        verify(userRepository, times(1)).findByEmail("noexiste@test.com");
    }

    /**
     * TC-007: Login con contraseña incorrecta
     * <p><b>Requisito:</b> RF2 - Manejo de login con contraseña incorrecta</p>
     * <p><b>Entrada:</b> Contraseña incorrecta</p>
     * <p><b>Salida esperada:</b> Optional vacío</p>
     */
    @Test
    void testLoginWithWrongPassword() {
        when(userRepository.findByEmail("juan@test.com")).thenReturn(Optional.of(testUser));
        testUser.setPassword("hashedPassword123");

        Optional<User> result = userService.login("juan@test.com", "WrongPassword");

        assertTrue(result.isEmpty());
        verify(userRepository, times(1)).findByEmail("juan@test.com");
    }

    /**
     * TC-008: Actualizar usuario inexistente (excepción)
     * <p><b>Requisito:</b> RF2 - Manejo de error al actualizar usuario inexistente</p>
     * <p><b>Entrada:</b> ID = 999 (usuario inexistente)</p>
     * <p><b>Salida esperada:</b> Excepción RuntimeException</p>
     */
    @Test
    void testUpdateUserNotFound() {
        when(userRepository.findById(999L)).thenReturn(Optional.empty());

        assertThrows(RuntimeException.class, () -> {
            userService.update(testUser, 999L);
        });

        verify(userRepository, never()).save(any(User.class));
    }

    /**
     * TC-009: Actualizar usuario con campos nulos
     * <p><b>Requisito:</b> RF2 - Actualización con campos nulos no debe sobrescribir</p>
     * <p><b>Entrada:</b> Usuario con campos nulos</p>
     * <p><b>Salida esperada:</b> Los campos nulos no sobrescriben los existentes</p>
     */
    @Test
    void testUpdateUserWithNullFields() {
        when(userRepository.findById(1L)).thenReturn(Optional.of(testUser));

        User nullUser = new User();
        nullUser.setFullName(null);
        nullUser.setEmail(null);

        userService.update(nullUser, 1L);

        verify(userRepository, times(1)).save(testUser);
    }

    /**
     * TC-010: Login con email vacío
     * <p><b>Requisito:</b> RF2 - Validación de campos vacíos en login</p>
     * <p><b>Entrada:</b> Email vacío</p>
     * <p><b>Salida esperada:</b> Optional vacío</p>
     */
    @Test
    void testLoginWithEmptyEmail() {
        Optional<User> result = userService.login("", "Password123");
        assertTrue(result.isEmpty());
    }

    // ================================================================
    // ========== PRUEBAS PARA RF3: RECUPERAR CONTRASEÑA ==========
    // ================================================================

    /**
     * TC-011: Recuperar contraseña - email existente
     * <p><b>Requisito:</b> RF3 - El sistema debe permitir recuperar contraseña por email</p>
     * <p><b>Entrada:</b> Email registrado en el sistema</p>
     * <p><b>Salida esperada:</b> Objeto User con los datos del usuario</p>
     */
    @Test
    void testFindByEmailExistente() {
        when(userRepository.findByEmail("juan@test.com")).thenReturn(Optional.of(testUser));

        User foundUser = userService.findByEmail("juan@test.com");

        assertNotNull(foundUser);
        assertEquals("juan@test.com", foundUser.getEmail());
        verify(userRepository, times(1)).findByEmail("juan@test.com");
    }

    /**
     * TC-012: Recuperar contraseña - email no registrado
     * <p><b>Requisito:</b> RF3 - Manejo de email no registrado</p>
     * <p><b>Entrada:</b> Email NO registrado en el sistema</p>
     * <p><b>Salida esperada:</b> null (usuario no encontrado)</p>
     */
    @Test
    void testFindByEmailNoExistente() {
        when(userRepository.findByEmail("noexiste@test.com")).thenReturn(Optional.empty());

        User foundUser = userService.findByEmail("noexiste@test.com");

        assertNull(foundUser);
        verify(userRepository, times(1)).findByEmail("noexiste@test.com");
    }

    /**
     * TC-013: Buscar usuario por email nulo
     * <p><b>Requisito:</b> RF3 - Manejo de email nulo</p>
     * <p><b>Entrada:</b> Email = null</p>
     * <p><b>Salida esperada:</b> null</p>
     */
    @Test
    void testFindByEmailNull() {
        User foundUser = userService.findByEmail(null);
        assertNull(foundUser);
    }

    // ================================================================
    // ========== PRUEBAS PARA RF4: ADMIN PROMUEVE USUARIOS ==========
    // ================================================================

    /**
     * TC-014: ADMIN promueve CLIENTE a ADMIN
     * <p><b>Requisito:</b> RF4 - El sistema debe permitir al ADMIN promover usuarios</p>
     * <p><b>Entrada:</b> ID de un usuario CLIENTE, nuevo rol ADMIN</p>
     * <p><b>Salida esperada:</b> Usuario actualizado con rol ADMIN</p>
     */
    @Test
    void testUpdateUserRoleToAdmin() {
        User clientUser = new User();
        clientUser.setRole(Role.CUSTOMER);

        when(userRepository.findById(1L)).thenReturn(Optional.of(clientUser));
        when(userRepository.save(any(User.class))).thenReturn(clientUser);

        clientUser.setRole(Role.ADMIN);
        userService.update(clientUser, 1L);

        assertEquals(Role.ADMIN, clientUser.getRole());
        verify(userRepository, times(1)).save(clientUser);
    }

    /**
     * TC-015: Buscar usuario por ID existente
     * <p><b>Requisito:</b> RF2 - El sistema debe permitir buscar usuarios por ID</p>
     * <p><b>Entrada:</b> ID = 1 (usuario existente)</p>
     * <p><b>Salida esperada:</b> Optional con los datos del usuario</p>
     */
    @Test
    void testFindByIdSuccess() {
        when(userRepository.findById(1L))
                .thenReturn(Optional.of(testUser));

        Optional<User> result = userService.findById(1L);

        assertTrue(result.isPresent());
        assertEquals("Juan Perez", result.get().getFullName());

        verify(userRepository, times(1)).findById(1L);
    }

    /**
     * TC-016: Guardar usuario con contraseña encriptada
     * <p><b>Requisito:</b> RF1 - El sistema debe almacenar contraseñas encriptadas</p>
     * <p><b>Entrada:</b> Usuario válido con contraseña en texto plano</p>
     * <p><b>Salida esperada:</b> Usuario guardado con contraseña diferente al texto original</p>
     */
    @Test
    void testSaveUserEncryptsPassword() {
        when(userRepository.save(any(User.class)))
                .thenReturn(testUser);

        User savedUser = userService.save(testUser);

        assertNotNull(savedUser);
        assertNotEquals("Password123", savedUser.getPassword());

        verify(userRepository, times(1)).save(any(User.class));
    }

    /**
     * TC-017: Login exitoso
     * <p><b>Requisito:</b> RF2 - El sistema debe permitir autenticación válida</p>
     * <p><b>Entrada:</b> Email y contraseña correctos</p>
     * <p><b>Salida esperada:</b> Optional con el usuario autenticado</p>
     */
    @Test
    void testLoginSuccess() {
        String hashedPassword =
                userService.hashPassword("Password123");

        testUser.setPassword(hashedPassword);

        when(userRepository.findByEmail("juan@test.com"))
                .thenReturn(Optional.of(testUser));

        Optional<User> result =
                userService.login("juan@test.com", "Password123");

        assertTrue(result.isPresent());
        assertEquals("juan@test.com",
                result.get().getEmail());

        verify(userRepository, times(1))
                .findByEmail("juan@test.com");
    }

    /**
     * TC-018: Generar contraseña aleatoria
     * <p><b>Requisito:</b> RF3 - El sistema debe generar contraseñas temporales</p>
     * <p><b>Entrada:</b> Ninguna</p>
     * <p><b>Salida esperada:</b> Contraseña aleatoria de 8 caracteres</p>
     */
    @Test
    void testGenerateRandomPassword() {
        String password =
                userService.generateRandomPassword();

        assertNotNull(password);
        assertEquals(8, password.length());
    }

    /**
     * TC-019: Eliminar usuario existente
     * <p><b>Requisito:</b> RF2 - El sistema debe permitir eliminar usuarios</p>
     * <p><b>Entrada:</b> ID = 1 (usuario existente)</p>
     * <p><b>Salida esperada:</b> Usuario eliminado correctamente</p>
     */
    @Test
    void testDeleteUserSuccess() {
        doNothing().when(userRepository)
                .deleteById(1L);

        assertDoesNotThrow(() ->
                userService.delete(1L));

        verify(userRepository, times(1))
                .deleteById(1L);
    }
}