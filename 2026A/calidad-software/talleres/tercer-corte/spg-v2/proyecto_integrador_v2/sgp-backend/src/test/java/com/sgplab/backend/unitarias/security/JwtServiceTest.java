package com.sgplab.backend.unitarias.security;

import com.sgplab.backend.model.entity.Usuario;
import com.sgplab.backend.model.enums.Rol;
import com.sgplab.backend.security.JwtService;
import io.jsonwebtoken.Claims;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pruebas unitarias para {@link JwtService}.
 */
class JwtServiceTest {

    private static final String SECRET = "testsecretkeyonlyfortestprofilemustbelongenough256bitsabcdefghijklmnop";
    private static final long EXPIRATION_MS = 3600000L;
    private static final String ISSUER = "sgp-lab-test";

    private JwtService jwtService;
    private Usuario usuario;

    @BeforeEach
    void setUp() {
        jwtService = new JwtService(SECRET, EXPIRATION_MS, ISSUER);
        usuario = new Usuario();
        usuario.setId(7L);
        usuario.setEmail("user@test.com");
        usuario.setNombre("Test User");
        usuario.setRol(Rol.ADMINISTRADOR);
    }

    @Test
    @DisplayName("generateToken produce un JWT no nulo y valido")
    void generateToken_OK() {
        String token = jwtService.generateToken(usuario);
        assertNotNull(token);
        assertFalse(token.isBlank());
        assertTrue(jwtService.isValid(token));
    }

    @Test
    @DisplayName("extractEmail recupera el subject correctamente")
    void extractEmail_OK() {
        String token = jwtService.generateToken(usuario);
        assertEquals("user@test.com", jwtService.extractEmail(token));
    }

    @Test
    @DisplayName("parseClaims contiene los claims uid, rol y nombre")
    void parseClaims_ContieneClaims() {
        String token = jwtService.generateToken(usuario);
        Claims claims = jwtService.parseClaims(token);
        assertNotNull(claims);
        assertEquals(7, ((Number) claims.get(JwtService.CLAIM_USER_ID)).intValue());
        assertEquals("ADMINISTRADOR", claims.get(JwtService.CLAIM_ROL));
        assertEquals("Test User", claims.get(JwtService.CLAIM_NOMBRE));
        assertEquals(ISSUER, claims.getIssuer());
    }

    @Test
    @DisplayName("isValid retorna false para token alterado")
    void isValid_TokenAlterado() {
        String token = jwtService.generateToken(usuario);
        String alterado = token.substring(0, token.length() - 5) + "xxxxx";
        assertFalse(jwtService.isValid(alterado));
    }

    @Test
    @DisplayName("parseClaims retorna null para token alterado")
    void parseClaims_TokenAlterado() {
        assertNull(jwtService.parseClaims("no.es.un.jwt"));
    }

    @Test
    @DisplayName("extractEmail retorna null para token invalido")
    void extractEmail_TokenInvalido() {
        assertNull(jwtService.extractEmail("garbage"));
    }

    @Test
    @DisplayName("isValid retorna false si token expiro")
    void isValid_TokenExpirado() throws Exception {
        JwtService cortoVida = new JwtService(SECRET, 1L, ISSUER);
        String token = cortoVida.generateToken(usuario);
        Thread.sleep(10);
        assertFalse(cortoVida.isValid(token));
    }

    @Test
    @DisplayName("Constructor rechaza secret demasiado corto")
    void constructor_SecretCorto() {
        assertThrows(IllegalStateException.class,
                () -> new JwtService("corto", EXPIRATION_MS, ISSUER));
    }

    @Test
    @DisplayName("getExpirationMs devuelve el valor configurado")
    void getExpirationMs_OK() {
        assertEquals(EXPIRATION_MS, jwtService.getExpirationMs());
    }
}
