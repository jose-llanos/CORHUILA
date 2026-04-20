package com.example;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Suite de pruebas unitarias para la clase {@link Biblioteca}.
 *
 * <p>Corresponde al <b>PASO 1 (RED)</b> del ciclo TDD: los tests fueron
 * escritos antes de que existiera la implementación de {@code Biblioteca.java},
 * por lo que inicialmente todos fallaban. Una vez creada la implementación
 * mínima (PASO 2, GREEN), todos los casos deben pasar sin modificar estos
 * tests.</p>
 *
 * <p>La suite cubre las siguientes áreas:</p>
 * <ul>
 *   <li><b>Creación:</b> nombre de la biblioteca y estado inicial vacío.</li>
 *   <li><b>Agregar libros:</b> casos válidos, libro nulo e ISBN duplicado.</li>
 *   <li><b>Búsqueda por título:</b> exacta, parcial y sin resultados.</li>
 *   <li><b>Búsqueda por autor:</b> exacta, parcial y sin resultados.</li>
 *   <li><b>Préstamo:</b> libro disponible, ya prestado e inexistente.</li>
 *   <li><b>Devolución:</b> libro prestado, no prestado e inexistente.</li>
 *   <li><b>Disponibilidad:</b> filtrado de libros disponibles tras un préstamo.</li>
 * </ul>
 *
 * @author com.example
 * @version 1.0
 * @see Biblioteca
 * @see Libro
 */
@DisplayName("Pruebas de la clase Biblioteca")
class BibliotecaTest {

    /**
     * Instancia de {@link Biblioteca} reinicializada antes de cada test
     * mediante {@link #setUp()}.
     */
    private Biblioteca biblioteca;

    /** Libro de prueba: <em>Cien años de soledad</em>, ISBN-001. */
    private Libro libro1;

    /** Libro de prueba: <em>El amor en los tiempos del cólera</em>, ISBN-002. */
    private Libro libro2;

    /** Libro de prueba: <em>1984</em>, ISBN-003. */
    private Libro libro3;

    // ----------------------------------------------------------------
    // Fixture
    // ----------------------------------------------------------------

    /**
     * Inicializa el fixture de prueba antes de cada caso de test.
     *
     * <p>Crea una {@link Biblioteca} vacía y tres instancias de {@link Libro}
     * con datos válidos y distintos ISBNs.</p>
     */
    @BeforeEach
    void setUp() {
        biblioteca = new Biblioteca("Biblioteca Central");

        libro1 = new Libro("Cien años de soledad", "Gabriel García Márquez", "ISBN-001");
        libro2 = new Libro("El amor en los tiempos del cólera", "Gabriel García Márquez", "ISBN-002");
        libro3 = new Libro("1984", "George Orwell", "ISBN-003");
    }

    // ----------------------------------------------------------------
    // Tests de creación
    // ----------------------------------------------------------------

    /**
     * Verifica que el constructor asigne el nombre correctamente y que
     * el objeto resultante no sea {@code null}.
     */
    @Test
    @DisplayName("Debe crear una biblioteca con nombre")
    void debeCrearBibliotecaConNombre() {
        assertNotNull(biblioteca);
        assertEquals("Biblioteca Central", biblioteca.getNombre());
    }

    /**
     * Verifica que una biblioteca recién creada no contenga ningún libro
     * en su catálogo.
     */
    @Test
    @DisplayName("Una biblioteca nueva debe estar vacía")
    void bibliotecaNuevaDebeEstarVacia() {
        assertEquals(0, biblioteca.getTotalLibros());
    }

    // ----------------------------------------------------------------
    // Tests de agregar libros
    // ----------------------------------------------------------------

    /**
     * Verifica que al agregar un libro válido el total del catálogo
     * incremente en uno.
     */
    @Test
    @DisplayName("Debe agregar un libro correctamente")
    void debeAgregarUnLibro() {
        biblioteca.agregarLibro(libro1);
        assertEquals(1, biblioteca.getTotalLibros());
    }

    /**
     * Verifica que es posible agregar varios libros de forma consecutiva
     * y que el total refleja correctamente la cantidad añadida.
     */
    @Test
    @DisplayName("Debe agregar múltiples libros")
    void debeAgregarMultiplesLibros() {
        biblioteca.agregarLibro(libro1);
        biblioteca.agregarLibro(libro2);
        biblioteca.agregarLibro(libro3);
        assertEquals(3, biblioteca.getTotalLibros());
    }

    /**
     * Verifica que intentar agregar {@code null} al catálogo lanza
     * una {@link IllegalArgumentException}.
     */
    @Test
    @DisplayName("No debe agregar un libro nulo")
    void noDebeAgregarLibroNulo() {
        assertThrows(IllegalArgumentException.class, () -> biblioteca.agregarLibro(null));
    }

    /**
     * Verifica que no es posible agregar dos libros con el mismo ISBN.
     * Se espera una {@link IllegalArgumentException} al intentar agregar
     * el duplicado.
     */
    @Test
    @DisplayName("No debe agregar un libro con ISBN duplicado")
    void noDebeAgregarLibroConIsbnDuplicado() {
        biblioteca.agregarLibro(libro1);
        Libro duplicado = new Libro("Otro título", "Otro autor", "ISBN-001");
        assertThrows(IllegalArgumentException.class, () -> biblioteca.agregarLibro(duplicado));
    }

    // ----------------------------------------------------------------
    // Tests de buscar por título
    // ----------------------------------------------------------------

    /**
     * Verifica que la búsqueda por título exacto retorna exactamente un
     * resultado con el título esperado.
     */
    @Test
    @DisplayName("Debe encontrar un libro por título exacto")
    void debeEncontrarLibroPorTituloExacto() {
        biblioteca.agregarLibro(libro1);
        List<Libro> resultado = biblioteca.buscarPorTitulo("Cien años de soledad");
        assertEquals(1, resultado.size());
        assertEquals("Cien años de soledad", resultado.get(0).getTitulo());
    }

    /**
     * Verifica que la búsqueda por título parcial, sin distinguir mayúsculas
     * ni minúsculas, retorna todos los libros cuyo título contiene la cadena
     * indicada.
     */
    @Test
    @DisplayName("Debe encontrar libros por título parcial (sin distinguir mayúsculas)")
    void debeEncontrarLibrosPorTituloParcial() {
        biblioteca.agregarLibro(libro1);
        biblioteca.agregarLibro(libro2);
        biblioteca.agregarLibro(libro3);

        List<Libro> resultado = biblioteca.buscarPorTitulo("amor");
        assertEquals(1, resultado.size());
        assertTrue(resultado.get(0).getTitulo().contains("amor"));
    }

    /**
     * Verifica que la búsqueda retorna una lista vacía cuando ningún libro
     * del catálogo coincide con el título buscado.
     */
    @Test
    @DisplayName("Debe retornar lista vacía si no encuentra por título")
    void debeRetornarListaVaciaSiNoEncuentraPorTitulo() {
        biblioteca.agregarLibro(libro1);
        List<Libro> resultado = biblioteca.buscarPorTitulo("Libro inexistente");
        assertTrue(resultado.isEmpty());
    }

    // ----------------------------------------------------------------
    // Tests de buscar por autor
    // ----------------------------------------------------------------

    /**
     * Verifica que la búsqueda por nombre de autor exacto retorna todos
     * los libros de dicho autor presentes en el catálogo.
     */
    @Test
    @DisplayName("Debe encontrar todos los libros de un autor")
    void debeEncontrarLibrosPorAutor() {
        biblioteca.agregarLibro(libro1);
        biblioteca.agregarLibro(libro2);
        biblioteca.agregarLibro(libro3);

        List<Libro> resultado = biblioteca.buscarPorAutor("Gabriel García Márquez");
        assertEquals(2, resultado.size());
    }

    /**
     * Verifica que la búsqueda por autor parcial, sin distinguir mayúsculas
     * ni minúsculas, retorna el libro del autor cuyo nombre contiene la
     * cadena indicada.
     */
    @Test
    @DisplayName("Debe buscar por autor parcial (sin distinguir mayúsculas)")
    void debeBuscarPorAutorParcial() {
        biblioteca.agregarLibro(libro1);
        biblioteca.agregarLibro(libro2);
        biblioteca.agregarLibro(libro3);

        List<Libro> resultado = biblioteca.buscarPorAutor("orwell");
        assertEquals(1, resultado.size());
        assertEquals("George Orwell", resultado.get(0).getAutor());
    }

    /**
     * Verifica que la búsqueda retorna una lista vacía cuando ningún libro
     * del catálogo corresponde al autor buscado.
     */
    @Test
    @DisplayName("Debe retornar lista vacía si no encuentra por autor")
    void debeRetornarListaVaciaSiNoEncuentraPorAutor() {
        biblioteca.agregarLibro(libro1);
        List<Libro> resultado = biblioteca.buscarPorAutor("Autor Desconocido");
        assertTrue(resultado.isEmpty());
    }

    // ----------------------------------------------------------------
    // Tests de prestar libros
    // ----------------------------------------------------------------

    /**
     * Verifica que prestar un libro disponible retorna {@code true} y que
     * el libro queda marcado como no disponible.
     */
    @Test
    @DisplayName("Debe prestar un libro disponible")
    void debePrestarUnLibroDisponible() {
        biblioteca.agregarLibro(libro1);
        boolean prestado = biblioteca.prestarLibro("ISBN-001");
        assertTrue(prestado);
        assertFalse(libro1.isDisponible());
    }

    /**
     * Verifica que intentar prestar un libro que ya se encuentra prestado
     * lanza una {@link IllegalStateException}.
     */
    @Test
    @DisplayName("No debe prestar un libro que ya está prestado")
    void noDebePrestarUnLibroYaPrestado() {
        biblioteca.agregarLibro(libro1);
        biblioteca.prestarLibro("ISBN-001");
        assertThrows(IllegalStateException.class, () -> biblioteca.prestarLibro("ISBN-001"));
    }

    /**
     * Verifica que intentar prestar un libro con un ISBN que no existe en
     * el catálogo lanza una {@link IllegalArgumentException}.
     */
    @Test
    @DisplayName("No debe prestar un libro que no existe")
    void noDebePrestarLibroInexistente() {
        assertThrows(IllegalArgumentException.class, () -> biblioteca.prestarLibro("ISBN-999"));
    }

    // ----------------------------------------------------------------
    // Tests de devolver libros
    // ----------------------------------------------------------------

    /**
     * Verifica que devolver un libro previamente prestado retorna {@code true}
     * y que el libro queda nuevamente disponible.
     */
    @Test
    @DisplayName("Debe devolver un libro prestado")
    void debeDevolverUnLibroPrestado() {
        biblioteca.agregarLibro(libro1);
        biblioteca.prestarLibro("ISBN-001");
        boolean devuelto = biblioteca.devolverLibro("ISBN-001");
        assertTrue(devuelto);
        assertTrue(libro1.isDisponible());
    }

    /**
     * Verifica que intentar devolver un libro que no está prestado lanza
     * una {@link IllegalStateException}.
     */
    @Test
    @DisplayName("No debe devolver un libro que no está prestado")
    void noDebeDevolverLibroNoPrestado() {
        biblioteca.agregarLibro(libro1);
        assertThrows(IllegalStateException.class, () -> biblioteca.devolverLibro("ISBN-001"));
    }

    /**
     * Verifica que intentar devolver un libro con un ISBN que no existe en
     * el catálogo lanza una {@link IllegalArgumentException}.
     */
    @Test
    @DisplayName("No debe devolver un libro que no existe")
    void noDebeDevolverLibroInexistente() {
        assertThrows(IllegalArgumentException.class, () -> biblioteca.devolverLibro("ISBN-999"));
    }

    // ----------------------------------------------------------------
    // Tests de libros disponibles
    // ----------------------------------------------------------------

    /**
     * Verifica que {@link Biblioteca#getLibrosDisponibles()} retorna únicamente
     * los libros cuya disponibilidad es {@code true}, excluyendo los que
     * han sido prestados.
     */
    @Test
    @DisplayName("Debe listar solo los libros disponibles")
    void debeListarSoloLibrosDisponibles() {
        biblioteca.agregarLibro(libro1);
        biblioteca.agregarLibro(libro2);
        biblioteca.agregarLibro(libro3);
        biblioteca.prestarLibro("ISBN-001");

        List<Libro> disponibles = biblioteca.getLibrosDisponibles();
        assertEquals(2, disponibles.size());
        assertTrue(disponibles.stream().allMatch(Libro::isDisponible));
    }
}