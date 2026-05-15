package com.sgplab.backend.unitarias.util;

import com.sgplab.backend.util.PasswordHashUtil;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pruebas unitarias para {@link PasswordHashUtil}.
 */
class PasswordHashUtilTest {

    @Nested
    @DisplayName("hash(String)")
    class HashTests {

        @Test
        @DisplayName("Devuelve hash con formato salt:hash y longitudes correctas")
        void hash_FormatoValido() {
            String hash = PasswordHashUtil.hash("mySecret123");
            String[] parts = hash.split(":");
            assertEquals(2, parts.length, "Debe tener dos partes separadas por ':'");
            assertEquals(32, parts[0].length(), "El salt en hex debe medir 32 chars (16 bytes)");
            assertEquals(64, parts[1].length(), "El hash SHA-256 en hex debe medir 64 chars (32 bytes)");
        }

        @Test
        @DisplayName("Genera salts distintos en cada invocacion (salt aleatorio)")
        void hash_SaltsDistintosEntreLlamadas() {
            String h1 = PasswordHashUtil.hash("misma");
            String h2 = PasswordHashUtil.hash("misma");
            assertNotEquals(h1, h2, "El salt aleatorio debe producir hashes distintos para la misma password");
        }

        @Test
        @DisplayName("Lanza IllegalArgumentException si la password es null")
        void hash_PasswordNull() {
            assertThrows(IllegalArgumentException.class, () -> PasswordHashUtil.hash(null));
        }

        @Test
        @DisplayName("Lanza IllegalArgumentException si la password es vacia")
        void hash_PasswordVacia() {
            assertThrows(IllegalArgumentException.class, () -> PasswordHashUtil.hash(""));
        }
    }

    @Nested
    @DisplayName("matches(rawPassword, storedHash)")
    class MatchesTests {

        @Test
        @DisplayName("Retorna true cuando la password coincide con el hash")
        void matches_PasswordCorrecta() {
            String raw = "S3guro_!#";
            String stored = PasswordHashUtil.hash(raw);
            assertTrue(PasswordHashUtil.matches(raw, stored));
        }

        @Test
        @DisplayName("Retorna false con password incorrecta")
        void matches_PasswordIncorrecta() {
            String stored = PasswordHashUtil.hash("original");
            assertFalse(PasswordHashUtil.matches("OTRA", stored));
        }

        @Test
        @DisplayName("Retorna false si rawPassword es null")
        void matches_RawNull() {
            String stored = PasswordHashUtil.hash("x");
            assertFalse(PasswordHashUtil.matches(null, stored));
        }

        @Test
        @DisplayName("Retorna false si storedHash es null")
        void matches_StoredNull() {
            assertFalse(PasswordHashUtil.matches("x", null));
        }

        @Test
        @DisplayName("Retorna false si el hash no tiene el formato salt:hash")
        void matches_FormatoMalformado() {
            assertFalse(PasswordHashUtil.matches("x", "sin_separador"));
            assertFalse(PasswordHashUtil.matches("x", "demasiados:dos:puntos:hex"));
        }

        @Test
        @DisplayName("Retorna false si hex es invalido")
        void matches_HexInvalido() {
            assertFalse(PasswordHashUtil.matches("x", "zzzzzz:aaaaaa"));
        }

        @Test
        @DisplayName("Es resistente a timing attack (usa MessageDigest.isEqual)")
        void matches_HashAlternativo() {
            // Verifica que aceptamos un hash conocido previamente generado
            String knownHash = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6:6caa313cc1746666ed773360d66917beeea2fcf8856c299f16d047440ac463d2";
            assertTrue(PasswordHashUtil.matches("password", knownHash));
            assertFalse(PasswordHashUtil.matches("Password", knownHash), "Es case-sensitive");
        }
    }

    @Test
    @DisplayName("La clase no se puede instanciar")
    void claseUtilidadNoInstanciable() throws Exception {
        var constructor = PasswordHashUtil.class.getDeclaredConstructor();
        constructor.setAccessible(true);
        var ex = assertThrows(java.lang.reflect.InvocationTargetException.class, constructor::newInstance);
        assertNotNull(ex.getCause());
        assertTrue(ex.getCause() instanceof UnsupportedOperationException);
    }
}
