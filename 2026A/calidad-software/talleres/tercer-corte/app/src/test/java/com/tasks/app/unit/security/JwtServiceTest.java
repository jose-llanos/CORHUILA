package com.tasks.app.unit.security;

import com.tasks.app.security.JwtService;
import com.tasks.app.entity.User;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import javax.crypto.SecretKey;
import java.time.LocalDateTime;
import java.util.Date;

import static org.junit.jupiter.api.Assertions.*;

/*
 * Pruebas unitarias de JwtService.
 *
 * A diferencia de los tests de servicio, aquí no usamos mocks.
 * Instanciamos JwtService directamente y le inyectamos
 * un secret de prueba.
 *
 * El secreto es Base64 de 32 bytes (256 bits), que es el mínimo
 * que exige el algoritmo HS256.
 */
@DisplayName("TU-05 — JwtService: Seguridad y Tokens JWT")
public class JwtServiceTest {

    // Clave de prueba: Base64 de 32 bytes (32 caracteres 'a' = 0x61 repetidos)
    // No usar esta clave en producción.
    private static final String SECRET_PRUEBA =
            "YWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWE=";

    private JwtService jwtService;
    private User usuario;

    @BeforeEach
    void prepararDatos() {
        // Creamos el servicio e inyectamos el secret de prueba con reflexión
        jwtService = new JwtService();
        ReflectionTestUtils.setField(jwtService, "secret", SECRET_PRUEBA);

        usuario = User.builder()
                .id(1L)
                .username("juan")
                .email("juan@mail.com")
                .password("hash")
                .createdAt(LocalDateTime.now())
                .build();
    }

    // =========================================================
    // TU05-01 — Generar token
    // =========================================================

    @Test
    @DisplayName("TU05-01: El token generado contiene el username como 'subject'")
    void generarToken_contieneUsernameComoSubject() {
        // Cuando: se genera el token para el usuario
        String token = jwtService.generateToken(usuario);

        // Entonces: el token no está vacío y el subject es el username
        assertNotNull(token);
        assertFalse(token.isEmpty());
        String subject = jwtService.extractUsername(token);
        assertEquals("juan", subject);
    }

    // =========================================================
    // TU05-02 — Validar token válido
    // =========================================================

    @Test
    @DisplayName("TU05-02: Un token recién emitido es considerado válido")
    void validarToken_tokenRecienGenerado_esValido() {
        // Dado: un token recién creado
        String token = jwtService.generateToken(usuario);

        // Cuando: se verifica su validez
        boolean esValido = jwtService.isTokenValid(token);

        // Entonces: el token es válido
        assertTrue(esValido);
    }

    // =========================================================
    // TU05-03 — Token expirado
    // =========================================================

    @Test
    @DisplayName("TU05-03: Un token expirado NO es válido (isTokenValid retorna false)")
    void validarToken_tokenExpirado_esInvalido() {
        // Dado: construimos un token con fecha de expiración en el pasado
        SecretKey clave = Keys.hmacShaKeyFor(Decoders.BASE64.decode(SECRET_PRUEBA));
        String tokenExpirado = Jwts.builder()
                .subject("juan")
                .issuedAt(new Date(0))        // emitido en 1970
                .expiration(new Date(1))      // expiró en 1970 también
                .signWith(clave)
                .compact();

        // Cuando: se verifica el token vencido
        boolean esValido = jwtService.isTokenValid(tokenExpirado);

        // Entonces: el servicio lo rechaza (retorna false, no lanza excepción)
        assertFalse(esValido);
    }

    // =========================================================
    // TU05-04 — Token con firma alterada
    // =========================================================

    @Test
    @DisplayName("TU05-04: Un token con firma alterada NO es válido (isTokenValid retorna false)")
    void validarToken_firmaAlterada_esInvalido() {
        // Dado: generamos un token válido y le cambiamos el último carácter
        String tokenOriginal = jwtService.generateToken(usuario);
        // Alteramos el token sumando o restando un carácter al final
        char ultimoCaracter = tokenOriginal.charAt(tokenOriginal.length() - 1);
        char caracterAlterado = (ultimoCaracter == 'A') ? 'B' : 'A';
        String tokenAlterado = tokenOriginal.substring(0, tokenOriginal.length() - 1) + caracterAlterado;

        // Cuando: se verifica el token adulterado
        boolean esValido = jwtService.isTokenValid(tokenAlterado);

        // Entonces: el servicio lo rechaza
        assertFalse(esValido);
    }

    // =========================================================
    // TU05-05 — Extraer username
    // =========================================================

    @Test
    @DisplayName("TU05-05: extractUsername retorna el username original del token")
    void extraerUsername_tokenValido_retornaUsernameOriginal() {
        // Dado: un token generado para "juan"
        String token = jwtService.generateToken(usuario);

        // Cuando: se extrae el username
        String usernameExtraido = jwtService.extractUsername(token);

        // Entonces: es exactamente "juan"
        assertEquals("juan", usernameExtraido);
    }
}