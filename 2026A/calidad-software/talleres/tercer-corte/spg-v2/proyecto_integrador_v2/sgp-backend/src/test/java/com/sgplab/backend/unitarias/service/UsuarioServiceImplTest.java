package com.sgplab.backend.unitarias.service;

import com.sgplab.backend.dto.request.UsuarioRequest;
import com.sgplab.backend.dto.response.UsuarioResponse;
import com.sgplab.backend.exception.BusinessRuleException;
import com.sgplab.backend.exception.DuplicateResourceException;
import com.sgplab.backend.exception.ResourceNotFoundException;
import com.sgplab.backend.model.entity.Usuario;
import com.sgplab.backend.model.enums.EstadoUsuario;
import com.sgplab.backend.model.enums.Rol;
import com.sgplab.backend.repository.IUsuarioRepository;
import com.sgplab.backend.service.impl.UsuarioServiceImpl;
import com.sgplab.backend.util.PasswordHashUtil;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Pruebas unitarias para {@link UsuarioServiceImpl}.
 */
@ExtendWith(MockitoExtension.class)
class UsuarioServiceImplTest {

    @Mock
    private IUsuarioRepository usuarioRepository;

    @InjectMocks
    private UsuarioServiceImpl usuarioService;

    private UsuarioRequest validRequest;
    private Usuario usuarioExistente;

    @BeforeEach
    void setUp() {
        validRequest = new UsuarioRequest();
        validRequest.setNombre("Juan Perez");
        validRequest.setEmail("juan@x.com");
        validRequest.setPassword("password123");
        validRequest.setRol(Rol.CLIENTE);

        usuarioExistente = new Usuario();
        usuarioExistente.setId(1L);
        usuarioExistente.setNombre("Original");
        usuarioExistente.setEmail("original@x.com");
        usuarioExistente.setPasswordHash(PasswordHashUtil.hash("old"));
        usuarioExistente.setRol(Rol.CLIENTE);
        usuarioExistente.setEstado(EstadoUsuario.ACTIVO);
    }

    // ---------- crear ----------

    @Test
    @DisplayName("crear: caso valido persiste con password hasheada")
    void crear_OK() {
        when(usuarioRepository.existsByEmail("juan@x.com")).thenReturn(false);
        when(usuarioRepository.save(any(Usuario.class))).thenAnswer(inv -> {
            Usuario u = inv.getArgument(0);
            u.setId(42L);
            return u;
        });

        UsuarioResponse response = usuarioService.crear(validRequest);

        assertNotNull(response);
        assertEquals(42L, response.getId());
        assertEquals("juan@x.com", response.getEmail());

        ArgumentCaptor<Usuario> captor = ArgumentCaptor.forClass(Usuario.class);
        verify(usuarioRepository).save(captor.capture());
        Usuario guardado = captor.getValue();
        assertNotNull(guardado.getPasswordHash());
        assertTrue(PasswordHashUtil.matches("password123", guardado.getPasswordHash()),
                "El hash almacenado debe corresponder a la password en claro");
        assertFalse(guardado.getPasswordHash().contains("password123"),
                "El hash NO debe contener la password en claro");
    }

    @Test
    @DisplayName("crear: lanza BusinessRuleException si password es null")
    void crear_SinPassword() {
        validRequest.setPassword(null);
        assertThrows(BusinessRuleException.class, () -> usuarioService.crear(validRequest));
        verify(usuarioRepository, never()).save(any());
    }

    @Test
    @DisplayName("crear: lanza BusinessRuleException si password es blank")
    void crear_PasswordBlank() {
        validRequest.setPassword("   ");
        assertThrows(BusinessRuleException.class, () -> usuarioService.crear(validRequest));
    }

    @Test
    @DisplayName("crear: lanza DuplicateResourceException si el email ya existe")
    void crear_EmailDuplicado() {
        when(usuarioRepository.existsByEmail("juan@x.com")).thenReturn(true);
        assertThrows(DuplicateResourceException.class, () -> usuarioService.crear(validRequest));
        verify(usuarioRepository, never()).save(any());
    }

    // ---------- obtener ----------

    @Test
    @DisplayName("obtenerPorId: caso valido retorna el DTO")
    void obtenerPorId_OK() {
        when(usuarioRepository.findById(1L)).thenReturn(Optional.of(usuarioExistente));
        UsuarioResponse r = usuarioService.obtenerPorId(1L);
        assertEquals("original@x.com", r.getEmail());
    }

    @Test
    @DisplayName("obtenerPorId: lanza ResourceNotFoundException si no existe")
    void obtenerPorId_NoExiste() {
        when(usuarioRepository.findById(99L)).thenReturn(Optional.empty());
        assertThrows(ResourceNotFoundException.class, () -> usuarioService.obtenerPorId(99L));
    }

    @Test
    @DisplayName("obtenerTodos: retorna lista mapeada")
    void obtenerTodos_OK() {
        when(usuarioRepository.findAll()).thenReturn(List.of(usuarioExistente));
        List<UsuarioResponse> lista = usuarioService.obtenerTodos();
        assertEquals(1, lista.size());
        assertEquals("original@x.com", lista.get(0).getEmail());
    }

    // ---------- actualizar ----------

    @Test
    @DisplayName("actualizar: caso valido sin cambio de password conserva el hash original")
    void actualizar_SinPassword() {
        String hashAnterior = usuarioExistente.getPasswordHash();
        validRequest.setPassword(null);

        when(usuarioRepository.findById(1L)).thenReturn(Optional.of(usuarioExistente));
        when(usuarioRepository.save(any(Usuario.class))).thenAnswer(inv -> inv.getArgument(0));

        UsuarioResponse r = usuarioService.actualizar(1L, validRequest);

        assertEquals("juan@x.com", r.getEmail());
        assertEquals(hashAnterior, usuarioExistente.getPasswordHash(),
                "Si la password no viene, NO se debe cambiar el hash.");
    }

    @Test
    @DisplayName("actualizar: con password nueva la rehashea")
    void actualizar_ConPassword() {
        String hashAnterior = usuarioExistente.getPasswordHash();
        validRequest.setPassword("nuevaPwd!");

        when(usuarioRepository.findById(1L)).thenReturn(Optional.of(usuarioExistente));
        when(usuarioRepository.save(any(Usuario.class))).thenAnswer(inv -> inv.getArgument(0));

        usuarioService.actualizar(1L, validRequest);

        assertFalse(hashAnterior.equals(usuarioExistente.getPasswordHash()));
        assertTrue(PasswordHashUtil.matches("nuevaPwd!", usuarioExistente.getPasswordHash()));
    }

    @Test
    @DisplayName("actualizar: lanza ResourceNotFoundException si no existe")
    void actualizar_NoExiste() {
        when(usuarioRepository.findById(1L)).thenReturn(Optional.empty());
        assertThrows(ResourceNotFoundException.class,
                () -> usuarioService.actualizar(1L, validRequest));
    }

    @Test
    @DisplayName("actualizar: lanza DuplicateResourceException si el nuevo email pertenece a otro usuario")
    void actualizar_EmailDeOtro() {
        validRequest.setEmail("otro@x.com");
        when(usuarioRepository.findById(1L)).thenReturn(Optional.of(usuarioExistente));
        when(usuarioRepository.existsByEmail("otro@x.com")).thenReturn(true);

        assertThrows(DuplicateResourceException.class,
                () -> usuarioService.actualizar(1L, validRequest));
    }

    @Test
    @DisplayName("actualizar: si el email no cambia, no consulta existsByEmail")
    void actualizar_SinCambioEmail() {
        validRequest.setEmail(usuarioExistente.getEmail());
        when(usuarioRepository.findById(1L)).thenReturn(Optional.of(usuarioExistente));
        when(usuarioRepository.save(any(Usuario.class))).thenAnswer(inv -> inv.getArgument(0));

        usuarioService.actualizar(1L, validRequest);
        verify(usuarioRepository, never()).existsByEmail(any());
    }

    // ---------- eliminar ----------

    @Test
    @DisplayName("eliminar: caso valido invoca deleteById")
    void eliminar_OK() {
        when(usuarioRepository.existsById(1L)).thenReturn(true);
        usuarioService.eliminar(1L);
        verify(usuarioRepository).deleteById(1L);
    }

    @Test
    @DisplayName("eliminar: lanza ResourceNotFoundException si no existe")
    void eliminar_NoExiste() {
        when(usuarioRepository.existsById(1L)).thenReturn(false);
        assertThrows(ResourceNotFoundException.class, () -> usuarioService.eliminar(1L));
        verify(usuarioRepository, never()).deleteById(any());
    }
}
