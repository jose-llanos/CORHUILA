package com.example;


import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Suite de pruebas unitarias para la clase {@link Libro}.
 *
 * <p>Corresponde al <b>PASO 1 (RED)</b> del ciclo TDD: los tests fueron
 * escritos antes de que existiera la implementación de {@code Libro.java},
 * por lo que inicialmente todos fallaban. Una vez creada la implementación
 * mínima (PASO 2, GREEN), todos los casos deben pasar sin modificar estos
 * tests.</p>
 *
 * <p>La suite cubre cuatro áreas principales:</p>
 * <ul>
 *   <li><b>Creación y atributos:</b> verifica que el constructor asigne
 *       correctamente los campos y que se rechacen valores inválidos.</li>
 *   <li><b>Estado inicial:</b> comprueba que un libro recién creado esté
 *       disponible.</li>
 *   <li><b>Ciclo préstamo/devolución:</b> valida las transiciones de estado
 *       y las excepciones ante operaciones ilegales.</li>
 *   <li><b>Representación textual:</b> asegura que {@code toString()} exponga
 *       los datos relevantes del libro.</li>
 * </ul>
 *
 * @author com.example
 * @version 1.0
 * @see Libro
 */
@DisplayName("Pruebas de la clase Libro")
class LibroTest {

    /**
     * Instancia de {@link Libro} compartida y reinicializada antes de cada test
     * mediante {@link #setUp()}.
     */
    private Libro libro;

    // ----------------------------------------------------------------
    // Fixture
    // ----------------------------------------------------------------

    /**
     * Inicializa el fixture de prueba antes de cada caso de test.
     *
     * <p>Crea un {@link Libro} con datos válidos que sirve como punto de
     * partida neutral para todos los tests de esta clase.</p>
     */
    @BeforeEach
    void setUp() {
        libro = new Libro("El Quijote", "Miguel de Cervantes", "978-84-376-0494-7");
    }

    // ----------------------------------------------------------------
    // Tests de creación / atributos
    // ----------------------------------------------------------------

    /**
     * Verifica que el constructor asigne correctamente título, autor e ISBN,
     * y que el objeto resultante no sea {@code null}.
     */
    @Test
    @DisplayName("Debe crear un libro con título, autor e ISBN")
    void debeCrearLibroConAtributos() {
        assertNotNull(libro);
        assertEquals("El Quijote", libro.getTitulo());
        assertEquals("Miguel de Cervantes", libro.getAutor());
        assertEquals("978-84-376-0494-7", libro.getIsbn());
    }

    /**
     * Verifica que todo libro recién creado comience con disponibilidad
     * {@code true}, es decir, listo para ser prestado.
     */
    @Test
    @DisplayName("Un libro recién creado debe estar disponible")
    void libroNuevoDebeEstarDisponible() {
        assertTrue(libro.isDisponible());
    }

    /**
     * Verifica que el constructor rechace un título {@code null} o compuesto
     * únicamente por espacios en blanco, lanzando {@link IllegalArgumentException}.
     */
    @Test
    @DisplayName("No debe permitir título nulo o vacío")
    void noDebePermitirTituloNuloOVacio() {
        assertThrows(IllegalArgumentException.class,
                () -> new Libro(null, "Autor", "ISBN-001"));

        assertThrows(IllegalArgumentException.class,
                () -> new Libro("   ", "Autor", "ISBN-001"));
    }

    /**
     * Verifica que el constructor rechace un autor {@code null} o compuesto
     * únicamente por espacios en blanco, lanzando {@link IllegalArgumentException}.
     */
    @Test
    @DisplayName("No debe permitir autor nulo o vacío")
    void noDebePermitirAutorNuloOVacio() {
        assertThrows(IllegalArgumentException.class,
                () -> new Libro("Título", null, "ISBN-001"));

        assertThrows(IllegalArgumentException.class,
                () -> new Libro("Título", "  ", "ISBN-001"));
    }

    /**
     * Verifica que el constructor rechace un ISBN {@code null} o vacío,
     * lanzando {@link IllegalArgumentException}.
     */
    @Test
    @DisplayName("No debe permitir ISBN nulo o vacío")
    void noDebePermitirIsbnNuloOVacio() {
        assertThrows(IllegalArgumentException.class,
                () -> new Libro("Título", "Autor", null));

        assertThrows(IllegalArgumentException.class,
                () -> new Libro("Título", "Autor", ""));
    }

    // ----------------------------------------------------------------
    // Tests de estado: préstamo y devolución
    // ----------------------------------------------------------------

    /**
     * Verifica que al invocar {@link Libro#prestar()} el libro deje de estar
     * disponible ({@code isDisponible()} retorna {@code false}).
     */
    @Test
    @DisplayName("Debe marcarse como no disponible al prestarse")
    void debeMarcarseNoDisponibleAlPrestarse() {
        libro.prestar();
        assertFalse(libro.isDisponible());
    }

    /**
     * Verifica que no sea posible prestar un libro que ya se encuentra prestado.
     * Se espera una {@link IllegalStateException} en el segundo intento de préstamo.
     */
    @Test
    @DisplayName("No debe poder prestarse si ya está prestado")
    void noDebePoderPrestarseDoVeces() {
        libro.prestar();
        assertThrows(IllegalStateException.class, () -> libro.prestar());
    }

    /**
     * Verifica que tras un ciclo completo de préstamo y devolución el libro
     * recupere su disponibilidad ({@code isDisponible()} retorna {@code true}).
     */
    @Test
    @DisplayName("Debe volver a estar disponible al devolverse")
    void debeVolverAEstarDisponibleAlDevolverse() {
        libro.prestar();
        libro.devolver();
        assertTrue(libro.isDisponible());
    }

    /**
     * Verifica que no sea posible devolver un libro que no fue prestado previamente.
     * Se espera una {@link IllegalStateException}.
     */
    @Test
    @DisplayName("No debe poder devolverse si no está prestado")
    void noDebePoderDevolversesinEstarPrestado() {
        assertThrows(IllegalStateException.class, () -> libro.devolver());
    }

    // ----------------------------------------------------------------
    // Tests de representación
    // ----------------------------------------------------------------

    /**
     * Verifica que el resultado de {@link Libro#toString()} incluya como mínimo
     * el título y el nombre del autor del libro.
     */
    @Test
    @DisplayName("toString debe incluir título y autor")
    void toStringDebeIncluirTituloYAutor() {
        String representacion = libro.toString();
        assertTrue(representacion.contains("El Quijote"));
        assertTrue(representacion.contains("Miguel de Cervantes"));
    }
}