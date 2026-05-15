package com.sgplab.backend.security;

import com.sgplab.backend.model.entity.Usuario;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.JwtException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

/**
 * Servicio responsable de generar y validar JSON Web Tokens (JWT) firmados con HS256.
 *
 * <p>El token incluye como claims:
 * <ul>
 *   <li><b>sub</b>: email del usuario (identificador principal)</li>
 *   <li><b>uid</b>: id numerico del usuario</li>
 *   <li><b>rol</b>: rol del usuario (ADMINISTRADOR/CLIENTE)</li>
 *   <li><b>nombre</b>: nombre del usuario</li>
 * </ul>
 *
 * @author SGP LAB Team
 */
@Service
public class JwtService {

    private static final Logger log = LoggerFactory.getLogger(JwtService.class);

    public static final String CLAIM_USER_ID = "uid";
    public static final String CLAIM_ROL = "rol";
    public static final String CLAIM_NOMBRE = "nombre";

    private final SecretKey secretKey;
    private final long expirationMs;
    private final String issuer;

    /**
     * Construye el servicio con la configuracion del perfil activo.
     *
     * @param secret       clave secreta (>=32 bytes UTF-8 o base64) inyectada desde {@code sgp.security.jwt.secret}
     * @param expirationMs duracion del token en milisegundos
     * @param issuer       emisor del token
     */
    public JwtService(
            @Value("${sgp.security.jwt.secret}") String secret,
            @Value("${sgp.security.jwt.expiration-ms}") long expirationMs,
            @Value("${sgp.security.jwt.issuer}") String issuer) {
        byte[] keyBytes = decodeSecret(secret);
        this.secretKey = Keys.hmacShaKeyFor(keyBytes);
        this.expirationMs = expirationMs;
        this.issuer = issuer;
    }

    private byte[] decodeSecret(String secret) {
        // Intentar base64 primero; si falla, usar bytes UTF-8 (siempre que tengan longitud suficiente)
        try {
            byte[] decoded = Decoders.BASE64.decode(secret);
            if (decoded.length >= 32) {
                return decoded;
            }
        } catch (IllegalArgumentException ignored) {
            // Continuar con UTF-8
        }
        byte[] raw = secret.getBytes(StandardCharsets.UTF_8);
        if (raw.length < 32) {
            throw new IllegalStateException(
                    "El secreto JWT debe tener al menos 32 bytes (256 bits). Longitud actual: " + raw.length);
        }
        return raw;
    }

    /**
     * Genera un JWT para un usuario autenticado.
     *
     * @param usuario usuario autenticado (no nulo)
     * @return cadena JWT compacta firmada
     */
    public String generateToken(Usuario usuario) {
        Date now = new Date();
        Date expiry = new Date(now.getTime() + expirationMs);

        Map<String, Object> claims = new HashMap<>();
        claims.put(CLAIM_USER_ID, usuario.getId());
        claims.put(CLAIM_ROL, usuario.getRol().name());
        claims.put(CLAIM_NOMBRE, usuario.getNombre());

        return Jwts.builder()
                .subject(usuario.getEmail())
                .issuer(issuer)
                .issuedAt(now)
                .expiration(expiry)
                .claims(claims)
                .signWith(secretKey)
                .compact();
    }

    /**
     * Extrae el email (subject) del token.
     *
     * @param token JWT compacto
     * @return email del usuario o {@code null} si el token es invalido.
     */
    public String extractEmail(String token) {
        Claims claims = parseClaims(token);
        return claims != null ? claims.getSubject() : null;
    }

    /**
     * Extrae todos los claims del token.
     *
     * @param token JWT compacto
     * @return claims o {@code null} si el token no se pudo parsear.
     */
    public Claims parseClaims(String token) {
        try {
            return Jwts.parser()
                    .verifyWith(secretKey)
                    .build()
                    .parseSignedClaims(token)
                    .getPayload();
        } catch (JwtException | IllegalArgumentException ex) {
            log.debug("Token JWT invalido: {}", ex.getMessage());
            return null;
        }
    }

    /**
     * Indica si el token es vigente y bien firmado.
     *
     * @param token JWT compacto
     * @return {@code true} si es valido.
     */
    public boolean isValid(String token) {
        Claims claims = parseClaims(token);
        return claims != null && claims.getExpiration().after(new Date());
    }

    public long getExpirationMs() {
        return expirationMs;
    }
}
