package com.corhuila.unitest;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pruebas unitarias para la clase GestorEstudiantes.
 * Pruebas 7 – 16
 */
@DisplayName("Pruebas: GestorEstudiantes")
class GestorEstudiantesTest {

    private GestorEstudiantes gestor;

    @BeforeEach
    void setUp() {
        gestor = new GestorEstudiantes();
        gestor.agregar(new Estudiante("E01", "Laura Gómez",   20, 4.5));
        gestor.agregar(new Estudiante("E02", "Carlos Ruiz",   22, 2.5));
        gestor.agregar(new Estudiante("E03", "Andrea Torres", 19, 3.8));
    }

    // ──────────────────────────────────────────────
    // PRUEBA 7 — Agregar estudiante incrementa el contador
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 07 - Agregar estudiante incrementa el contador")
    void testAgregarEstudiante() {
        gestor.agregar(new Estudiante("E04", "Mario López", 23, 3.2));
        assertEquals(4, gestor.contarEstudiantes());
    }

    // ──────────────────────────────────────────────
    // PRUEBA 8 — Agregar ID duplicado lanza excepción
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 08 - ID duplicado lanza IllegalStateException")
    void testAgregarIdDuplicado() {
        Estudiante duplicado = new Estudiante("E01", "Otro Nombre", 25, 4.0);
        assertThrows(IllegalStateException.class,
                () -> gestor.agregar(duplicado));
    }

    // ──────────────────────────────────────────────
    // PRUEBA 9 — Buscar estudiante existente por ID
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 09 - Buscar estudiante existente retorna el objeto correcto")
    void testBuscarEstudianteExistente() {
        Optional<Estudiante> resultado = gestor.buscarPorId("E02");
        assertTrue(resultado.isPresent());
        assertEquals("Carlos Ruiz", resultado.get().getNombre());
    }

    // ──────────────────────────────────────────────
    // PRUEBA 10 — Buscar ID inexistente retorna vacío
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 10 - Buscar ID inexistente retorna Optional vacío")
    void testBuscarEstudianteNoExistente() {
        Optional<Estudiante> resultado = gestor.buscarPorId("X99");
        assertTrue(resultado.isEmpty(), "Debería retornar un Optional vacío");
    }

    // ──────────────────────────────────────────────
    // PRUEBA 11 — Eliminar estudiante existente
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 11 - Eliminar estudiante existente reduce el contador")
    void testEliminarEstudianteExistente() {
        gestor.eliminar("E03");
        assertEquals(2, gestor.contarEstudiantes());
        assertTrue(gestor.buscarPorId("E03").isEmpty());
    }

    // ──────────────────────────────────────────────
    // PRUEBA 12 — Eliminar ID inexistente lanza excepción
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 12 - Eliminar ID inexistente lanza IllegalArgumentException")
    void testEliminarEstudianteNoExistente() {
        assertThrows(IllegalArgumentException.class,
                () -> gestor.eliminar("X99"));
    }

    // ──────────────────────────────────────────────
    // PRUEBA 13 — Calcular promedio correcto
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 13 - Promedio de notas calculado correctamente")
    void testCalcularPromedio() {
        // (4.5 + 2.5 + 3.8) / 3 = 3.6̄
        double promedio = gestor.calcularPromedio();
        assertEquals(3.6, promedio, 0.01, "El promedio debería ser ~3.6");
    }

    // ──────────────────────────────────────────────
    // PRUEBA 14 — Promedio sin estudiantes lanza excepción
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 14 - Promedio sin estudiantes lanza IllegalStateException")
    void testPromedioSinEstudiantes() {
        GestorEstudiantes gestorVacio = new GestorEstudiantes();
        assertThrows(IllegalStateException.class,
                gestorVacio::calcularPromedio);
    }

    // ──────────────────────────────────────────────
    // PRUEBA 15 — Obtener lista de aprobados
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 15 - Solo Laura y Andrea están aprobadas")
    void testObtenerAprobados() {
        List<Estudiante> aprobados = gestor.obtenerAprobados();
        assertEquals(2, aprobados.size());
        assertTrue(aprobados.stream().allMatch(Estudiante::aprobado));
    }

    // ──────────────────────────────────────────────
    // PRUEBA 16 — Obtener el mejor estudiante
    // ──────────────────────────────────────────────
    @Test
    @DisplayName("Prueba 16 - El mejor estudiante es Laura con nota 4.5")
    void testObtenerMejorEstudiante() {
        Estudiante mejor = gestor.obtenerMejorEstudiante();
        assertEquals("E01", mejor.getId());
        assertEquals(4.5, mejor.getNota());
    }
}
