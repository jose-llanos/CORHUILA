package com.sgplab.backend.pruebasUnitarias.service;

import com.sgplab.backend.model.entity.Usuario;
import com.sgplab.backend.model.enums.Rol;
import com.sgplab.backend.repository.IUsuarioRepository;
import com.sgplab.backend.service.UsuarioService; // <-- Corrección: Usar la clase concreta
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
public class UsuarioServiceTest {

    @Mock
    private IUsuarioRepository usuarioRepository;

    // 1. Corrección: Usar la clase concreta UsuarioService para que Mockito pueda instanciarla
    @InjectMocks
    private UsuarioService usuarioService;

    @Test
    @DisplayName("SQA: Verificar validación de email inválido al crear usuario")
    void testCrearUsuario_EmailInvalido() {
        // Explicación: Valida que el servicio bloquee la creación y lance un IllegalArgumentException
        // si el objeto Usuario no contiene un email válido (nulo o sin '@').

        Usuario user = new Usuario();
        user.setEmail("correoInvalido.com"); // Falta el @

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            usuarioService.crearUsuario(user);
        });

        assertEquals("El email proporcionado no es válido.", exception.getMessage());
    }

    @Test
    @DisplayName("SQA: Verificar prevención de elevación de privilegios (Forzar rol CLIENTE)")
    void testCrearUsuario_PrevencionElevacionPrivilegios() {
        // Explicación: Valida que, aunque un atacante o usuario intente inyectar el rol de ADMINISTRADOR
        // en la petición, el servicio sobrescriba este valor y asigne el rol CLIENTE de forma segura.

        Usuario user = new Usuario();
        user.setEmail("usuario@valido.com"); // Email válido para pasar la primera validación
        user.setRol(Rol.ADMINISTRADOR); // Intento malicioso de crear un admin

        // Configuramos el mock para que devuelva el mismo objeto que se intentó guardar
        Mockito.when(usuarioRepository.save(any(Usuario.class)))
                .thenAnswer(invocation -> invocation.getArgument(0));

        Usuario resultado = usuarioService.crearUsuario(user);

        // Verificamos que el rol final haya sido forzado a CLIENTE
        assertEquals(Rol.CLIENTE, resultado.getRol(), "Seguridad: El servicio debe ignorar el rol enviado y asignar CLIENTE por defecto.");
        verify(usuarioRepository).save(user);
    }
}