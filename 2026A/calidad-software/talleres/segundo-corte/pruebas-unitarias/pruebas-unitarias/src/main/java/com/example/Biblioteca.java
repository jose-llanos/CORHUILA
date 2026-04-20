package com.example;



import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Representa una biblioteca que gestiona un catálogo de libros.
 *
 * <p>Permite agregar libros, realizar búsquedas por título o autor,
 * gestionar el ciclo de préstamo y devolución, y consultar el estado
 * general del catálogo.</p>
 *
 * <p>Esta clase fue desarrollada siguiendo la metodología TDD:</p>
 * <ul>
 *   <li><b>PASO 1 (RED):</b> Se escribieron los tests en {@code BibliotecaTest}
 *       antes de que existiera esta implementación.</li>
 *   <li><b>PASO 2 (GREEN):</b> Se escribió el mínimo código necesario para que
 *       todos los tests pasaran.</li>
 *   <li><b>PASO 3 (REFACTOR):</b> Se adoptaron Streams para las búsquedas y se
 *       extrajo el método privado {@link #buscarPorIsbn(String)} para eliminar
 *       duplicación de lógica.</li>
 * </ul>
 *
 * @author com.example
 * @version 1.0
 * @see Libro
 */
public class Biblioteca {

    /** Nombre identificador de la biblioteca. Inmutable tras la construcción. */
    private final String nombre;

    /**
     * Colección interna de libros que conforman el catálogo.
     * Cada elemento es único por ISBN.
     */
    private final List<Libro> catalogo;

    // ----------------------------------------------------------------
    // Constructor
    // ----------------------------------------------------------------

    /**
     * Crea una nueva biblioteca con el nombre indicado y un catálogo vacío.
     *
     * <p>Los espacios en blanco al inicio y al final del nombre se eliminan
     * automáticamente mediante {@link String#trim()}.</p>
     *
     * @param nombre nombre de la biblioteca; no puede ser {@code null} ni vacío
     * @throws IllegalArgumentException si {@code nombre} es {@code null} o contiene
     *                                  únicamente espacios en blanco
     */
    public Biblioteca(String nombre) {
        if (nombre == null || nombre.trim().isEmpty()) {
            throw new IllegalArgumentException("El nombre de la biblioteca no puede ser nulo o vacío.");
        }
        this.nombre   = nombre.trim();
        this.catalogo = new ArrayList<>();
    }

    // ----------------------------------------------------------------
    // Agregar libros
    // ----------------------------------------------------------------

    /**
     * Agrega un libro al catálogo de la biblioteca.
     *
     * <p>El ISBN del libro debe ser único dentro del catálogo; si ya existe
     * otro libro con el mismo ISBN (comparación sin distinguir mayúsculas),
     * la operación se rechaza.</p>
     *
     * @param libro el libro a agregar; no puede ser {@code null}
     * @throws IllegalArgumentException si {@code libro} es {@code null} o si su
     *                                  ISBN ya existe en el catálogo
     */
    public void agregarLibro(Libro libro) {
        if (libro == null) {
            throw new IllegalArgumentException("No se puede agregar un libro nulo.");
        }
        boolean isbnDuplicado = catalogo.stream()
                .anyMatch(l -> l.getIsbn().equalsIgnoreCase(libro.getIsbn()));
        if (isbnDuplicado) {
            throw new IllegalArgumentException(
                    "Ya existe un libro con el ISBN '" + libro.getIsbn() + "' en el catálogo."
            );
        }
        catalogo.add(libro);
    }

    // ----------------------------------------------------------------
    // Buscar por título
    // ----------------------------------------------------------------

    /**
     * Busca libros cuyo título contenga la cadena indicada, ignorando
     * mayúsculas y minúsculas.
     *
     * <p>Si {@code titulo} es {@code null} o está vacío, se retorna una
     * lista vacía sin lanzar excepción.</p>
     *
     * @param titulo texto a buscar dentro del título de los libros
     * @return lista (posiblemente vacía) de libros que coinciden con la búsqueda;
     *         nunca {@code null}
     */
    public List<Libro> buscarPorTitulo(String titulo) {
        if (titulo == null || titulo.trim().isEmpty()) {
            return new ArrayList<>();
        }
        String busqueda = titulo.trim().toLowerCase();
        return catalogo.stream()
                .filter(l -> l.getTitulo().toLowerCase().contains(busqueda))
                .collect(Collectors.toList());
    }

    // ----------------------------------------------------------------
    // Buscar por autor
    // ----------------------------------------------------------------

    /**
     * Busca libros cuyo autor contenga la cadena indicada, ignorando
     * mayúsculas y minúsculas.
     *
     * <p>Si {@code autor} es {@code null} o está vacío, se retorna una
     * lista vacía sin lanzar excepción.</p>
     *
     * @param autor texto a buscar dentro del nombre del autor de los libros
     * @return lista (posiblemente vacía) de libros que coinciden con la búsqueda;
     *         nunca {@code null}
     */
    public List<Libro> buscarPorAutor(String autor) {
        if (autor == null || autor.trim().isEmpty()) {
            return new ArrayList<>();
        }
        String busqueda = autor.trim().toLowerCase();
        return catalogo.stream()
                .filter(l -> l.getAutor().toLowerCase().contains(busqueda))
                .collect(Collectors.toList());
    }

    // ----------------------------------------------------------------
    // Prestar libro
    // ----------------------------------------------------------------

    /**
     * Presta el libro identificado por el ISBN proporcionado.
     *
     * <p>Delega en {@link Libro#prestar()} el cambio de estado. El libro
     * debe existir en el catálogo y estar disponible en el momento de la
     * llamada.</p>
     *
     * @param isbn ISBN del libro a prestar
     * @return {@code true} si el préstamo se realizó con éxito
     * @throws IllegalArgumentException si no existe ningún libro con ese ISBN
     * @throws IllegalStateException    si el libro ya se encuentra prestado
     */
    public boolean prestarLibro(String isbn) {
        Libro libro = buscarPorIsbn(isbn);
        libro.prestar();
        return true;
    }

    // ----------------------------------------------------------------
    // Devolver libro
    // ----------------------------------------------------------------

    /**
     * Registra la devolución del libro identificado por el ISBN proporcionado.
     *
     * <p>Delega en {@link Libro#devolver()} el cambio de estado. El libro
     * debe existir en el catálogo y estar actualmente prestado.</p>
     *
     * @param isbn ISBN del libro a devolver
     * @return {@code true} si la devolución se realizó con éxito
     * @throws IllegalArgumentException si no existe ningún libro con ese ISBN
     * @throws IllegalStateException    si el libro no se encuentra prestado
     */
    public boolean devolverLibro(String isbn) {
        Libro libro = buscarPorIsbn(isbn);
        libro.devolver();
        return true;
    }

    // ----------------------------------------------------------------
    // Consultas
    // ----------------------------------------------------------------

    /**
     * Retorna la lista de libros que actualmente están disponibles para préstamo.
     *
     * @return lista (posiblemente vacía) de libros con {@link Libro#isDisponible()}
     *         igual a {@code true}; nunca {@code null}
     */
    public List<Libro> getLibrosDisponibles() {
        return catalogo.stream()
                .filter(Libro::isDisponible)
                .collect(Collectors.toList());
    }

    /**
     * Retorna el número total de libros registrados en el catálogo,
     * independientemente de su disponibilidad.
     *
     * @return cantidad total de libros en el catálogo; {@code 0} si está vacío
     */
    public int getTotalLibros() {
        return catalogo.size();
    }

    /**
     * Retorna el nombre de la biblioteca.
     *
     * @return nombre de la biblioteca; nunca {@code null} ni vacío
     */
    public String getNombre() {
        return nombre;
    }

    // ----------------------------------------------------------------
    // Método privado reutilizable
    // ----------------------------------------------------------------

    /**
     * Busca un libro en el catálogo por su ISBN, sin distinguir mayúsculas
     * y minúsculas.
     *
     * <p>Método auxiliar extraído durante la fase de refactorización para
     * centralizar la lógica de búsqueda por ISBN y evitar duplicación entre
     * {@link #prestarLibro(String)} y {@link #devolverLibro(String)}.</p>
     *
     * @param isbn ISBN del libro a localizar
     * @return el {@link Libro} cuyo ISBN coincide con el proporcionado
     * @throws IllegalArgumentException si ningún libro del catálogo tiene ese ISBN
     */
    private Libro buscarPorIsbn(String isbn) {
        return catalogo.stream()
                .filter(l -> l.getIsbn().equalsIgnoreCase(isbn))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException(
                        "No se encontró ningún libro con el ISBN '" + isbn + "'."
                ));
    }

    // ----------------------------------------------------------------
    // Representación
    // ----------------------------------------------------------------

    /**
     * Retorna una representación textual de la biblioteca con su nombre
     * y el total de libros en catálogo.
     *
     * <p>Formato de ejemplo:</p>
     * <pre>
     * Biblioteca{nombre='Biblioteca Central', totalLibros=3}
     * </pre>
     *
     * @return cadena descriptiva de la biblioteca
     */
    @Override
    public String toString() {
        return String.format("Biblioteca{nombre='%s', totalLibros=%d}", nombre, catalogo.size());
    }
}