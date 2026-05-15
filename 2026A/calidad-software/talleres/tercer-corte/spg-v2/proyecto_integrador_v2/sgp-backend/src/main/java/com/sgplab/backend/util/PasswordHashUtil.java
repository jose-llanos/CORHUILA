package com.sgplab.backend.util;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.HexFormat;

/**
 * Utilidad para el hash y verificacion de contrasenas mediante SHA-256
 * con salt unico por contrasena.
 *
 * <p>Formato de almacenamiento: {@code saltHex:hashHex}, donde:
 * <ul>
 *   <li>{@code saltHex}: 16 bytes aleatorios (32 chars hex) generados con SecureRandom.</li>
 *   <li>{@code hashHex}: 32 bytes (64 chars hex) resultado de SHA-256(salt || password).</li>
 * </ul>
 *
 * <p><b>Importante:</b> El uso de salt unico por contrasena previene ataques con
 * rainbow tables. No obstante, SHA-256 es un hash rapido; para sistemas en
 * produccion con alto riesgo se recomienda BCrypt/Argon2 (cost-factor adaptativo).
 * El uso de SHA-256 en este proyecto cumple un requisito academico explicito.
 *
 * @author SGP LAB Team
 * @version 1.0.0
 */
public final class PasswordHashUtil {

    /** Algoritmo de hash empleado. */
    public static final String ALGORITHM = "SHA-256";

    /** Longitud en bytes del salt aleatorio. */
    public static final int SALT_LENGTH = 16;

    /** Separador entre salt y hash en la cadena almacenada. */
    public static final String SEPARATOR = ":";

    private static final SecureRandom SECURE_RANDOM = new SecureRandom();
    private static final HexFormat HEX = HexFormat.of();

    private PasswordHashUtil() {
        throw new UnsupportedOperationException("Clase de utilidad: no debe instanciarse.");
    }

    /**
     * Calcula el hash de una contrasena con un salt aleatorio fresco.
     *
     * @param rawPassword contrasena en claro; no puede ser {@code null} ni vacia.
     * @return cadena con formato {@code saltHex:hashHex} apta para persistencia.
     * @throws IllegalArgumentException si {@code rawPassword} es {@code null} o vacia.
     * @throws IllegalStateException si el algoritmo SHA-256 no esta disponible en la JVM.
     */
    public static String hash(String rawPassword) {
        if (rawPassword == null || rawPassword.isEmpty()) {
            throw new IllegalArgumentException("La contrasena no puede ser nula ni vacia.");
        }
        byte[] salt = new byte[SALT_LENGTH];
        SECURE_RANDOM.nextBytes(salt);
        byte[] hash = digest(salt, rawPassword);
        return HEX.formatHex(salt) + SEPARATOR + HEX.formatHex(hash);
    }

    /**
     * Verifica si una contrasena en claro coincide con un hash almacenado.
     *
     * @param rawPassword contrasena en claro a comprobar.
     * @param storedHash  cadena almacenada en formato {@code saltHex:hashHex}.
     * @return {@code true} si coinciden, {@code false} en cualquier otro caso
     *         (incluido entradas nulas o malformadas).
     */
    public static boolean matches(String rawPassword, String storedHash) {
        if (rawPassword == null || storedHash == null) {
            return false;
        }
        String[] parts = storedHash.split(SEPARATOR);
        if (parts.length != 2) {
            return false;
        }
        try {
            byte[] salt = HEX.parseHex(parts[0]);
            byte[] expected = HEX.parseHex(parts[1]);
            byte[] actual = digest(salt, rawPassword);
            return MessageDigest.isEqual(expected, actual);
        } catch (IllegalArgumentException ex) {
            return false;
        }
    }

    private static byte[] digest(byte[] salt, String rawPassword) {
        try {
            MessageDigest md = MessageDigest.getInstance(ALGORITHM);
            md.update(salt);
            return md.digest(rawPassword.getBytes(StandardCharsets.UTF_8));
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException("Algoritmo " + ALGORITHM + " no disponible en la JVM.", ex);
        }
    }
}
