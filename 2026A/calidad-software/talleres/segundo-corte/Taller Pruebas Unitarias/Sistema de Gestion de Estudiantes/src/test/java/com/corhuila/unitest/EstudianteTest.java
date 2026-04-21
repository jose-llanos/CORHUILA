package com.corhuila.unitest;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pruebas unitarias para la clase Estudiante.
 * Pruebas 1 – 6
 */
@DisplayName("Pruebas: Estudiante")
class EstudianteTest {

    // ──────────────────────────────────────────────
    // PRUEBA 1 — Crear estudiante con datos válidos
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 01 - Crear estudiante válido")
    void testCrearEstudianteValido() {
        Estudiante e = new Estudiante("E01", "Laura Gómez", 20, 4.5);

        assertAll(
                () -> assertEquals("E01",        e.getId()),
                () -> assertEquals("Laura Gómez", e.getNombre()),
                () -> assertEquals(20,            e.getEdad()),
                () -> assertEquals(4.5,           e.getNota())
        );
    }

    // ──────────────────────────────────────────────
    // PRUEBA 2 — ID vacío lanza excepción
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 02 - ID vacío lanza IllegalArgumentException")
    void testIdVacioLanzaExcepcion() {
        assertThrows(IllegalArgumentException.class,
                () -> new Estudiante("", "Carlos", 18, 3.5));
    }

    // ──────────────────────────────────────────────
    // PRUEBA 3 — Edad fuera de rango lanza excepción
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 03 - Edad fuera de rango lanza IllegalArgumentException")
    void testEdadFueraDeRango() {
        assertThrows(IllegalArgumentException.class,
                () -> new Estudiante("E02", "Pedro", 3, 4.0));
    }

    // ──────────────────────────────────────────────
    // PRUEBA 4 — Nota fuera de rango lanza excepción
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 04 - Nota mayor a 5.0 lanza IllegalArgumentException")
    void testNotaMayorAlMaximo() {
        assertThrows(IllegalArgumentException.class,
                () -> new Estudiante("E03", "Ana", 22, 6.0));
    }

    // ──────────────────────────────────────────────
    // PRUEBA 5 — Estudiante aprueba con nota >= 3.0
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 05 - Estudiante con nota 3.0 está aprobado")
    void testEstudianteAprobado() {
        Estudiante e = new Estudiante("E04", "Luis", 19, 3.0);
        assertTrue(e.aprobado(), "Con nota 3.0 el estudiante debería estar aprobado");
    }

    // ──────────────────────────────────────────────
    // PRUEBA 6 — Estudiante reprueba con nota < 3.0
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 06 - Estudiante con nota 2.9 está reprobado")
    void testEstudianteReprobado() {
        Estudiante e = new Estudiante("E05", "Sofía", 21, 2.9);
        assertFalse(e.aprobado(), "Con nota 2.9 el estudiante debería estar reprobado");
    }
}
