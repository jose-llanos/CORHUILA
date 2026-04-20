package com.example;



/**
 * Representa un libro dentro del catálogo de una biblioteca.
 *
 * <p>Cada libro se identifica por su título, autor e ISBN, y mantiene
 * un estado de disponibilidad que cambia según las operaciones de
 * préstamo y devolución. Todo libro comienza disponible al ser creado.</p>
 *
 * <p>Esta clase fue desarrollada siguiendo la metodología TDD:</p>
 * <ul>
 *   <li><b>PASO 1 (RED):</b> Se escribieron los tests en {@code LibroTest} antes de
 *       que existiera esta implementación.</li>
 *   <li><b>PASO 2 (GREEN):</b> Se escribió el mínimo código necesario para que
 *       todos los tests pasaran.</li>
 *   <li><b>PASO 3 (REFACTOR):</b> Se añadieron validaciones y se extrajo el método
 *       {@code validarCampo()} para evitar duplicación.</li>
 * </ul>
 *
 * @author com.example
 * @version 1.0
 * @see Biblioteca
 */
public class Libro {

    /** Título del libro. Inmutable tras la construcción. */
    private final String titulo;

    /** Nombre del autor del libro. Inmutable tras la construcción. */
    private final String autor;

    /**
     * Código ISBN que identifica de forma única al libro.
     * Inmutable tras la construcción.
     */
    private final String isbn;

    /**
     * Indica si el libro está disponible para préstamo.
     * {@code true} cuando el libro puede prestarse; {@code false} cuando ya
     * ha sido prestado y aún no ha sido devuelto.
     */
    private boolean disponible;

    // ----------------------------------------------------------------
    // Constructor
    // ----------------------------------------------------------------

    /**
     * Crea un nuevo libro con los datos proporcionados.
     *
     * <p>El libro se inicializa como disponible. Los espacios en blanco
     * al inicio y al final de cada campo son eliminados automáticamente
     * mediante {@link String#trim()}.</p>
     *
     * @param titulo título del libro; no puede ser {@code null} ni estar vacío
     * @param autor  nombre del autor; no puede ser {@code null} ni estar vacío
     * @param isbn   código ISBN del libro; no puede ser {@code null} ni estar vacío
     * @throws IllegalArgumentException si alguno de los parámetros es {@code null}
     *                                  o contiene únicamente espacios en blanco
     */
    public Libro(String titulo, String autor, String isbn) {
        validarCampo(titulo, "El título no puede ser nulo o vacío");
        validarCampo(autor,  "El autor no puede ser nulo o vacío");
        validarCampo(isbn,   "El ISBN no puede ser nulo o vacío");

        this.titulo     = titulo.trim();
        this.autor      = autor.trim();
        this.isbn       = isbn.trim();
        this.disponible = true;
    }

    // ----------------------------------------------------------------
    // Comportamiento: préstamo y devolución
    // ----------------------------------------------------------------

    /**
     * Marca el libro como prestado, cambiando su disponibilidad a {@code false}.
     *
     * <p>Solo puede prestarse un libro que actualmente esté disponible.
     * Intentar prestar un libro que ya se encuentra prestado lanzará una
     * excepción.</p>
     *
     * @throws IllegalStateException si el libro ya está prestado
     */
    public void prestar() {
        if (!disponible) {
            throw new IllegalStateException(
                    "El libro '" + titulo + "' ya está prestado y no puede prestarse de nuevo."
            );
        }
        disponible = false;
    }

    /**
     * Marca el libro como devuelto, restaurando su disponibilidad a {@code true}.
     *
     * <p>Solo puede devolverse un libro que actualmente se encuentre prestado.
     * Intentar devolver un libro que no fue prestado previamente lanzará una
     * excepción.</p>
     *
     * @throws IllegalStateException si el libro no está prestado en el momento
     *                               de invocar este método
     */
    public void devolver() {
        if (disponible) {
            throw new IllegalStateException(
                    "El libro '" + titulo + "' no está prestado, no se puede devolver."
            );
        }
        disponible = true;
    }

    // ----------------------------------------------------------------
    // Getters
    // ----------------------------------------------------------------

    /**
     * Retorna el título del libro.
     *
     * @return título del libro; nunca {@code null} ni vacío
     */
    public String getTitulo()  { return titulo; }

    /**
     * Retorna el nombre del autor del libro.
     *
     * @return autor del libro; nunca {@code null} ni vacío
     */
    public String getAutor()   { return autor; }

    /**
     * Retorna el código ISBN del libro.
     *
     * @return ISBN del libro; nunca {@code null} ni vacío
     */
    public String getIsbn()    { return isbn; }

    /**
     * Indica si el libro está disponible para ser prestado.
     *
     * @return {@code true} si el libro puede prestarse;
     *         {@code false} si ya se encuentra prestado
     */
    public boolean isDisponible() { return disponible; }

    // ----------------------------------------------------------------
    // Utilidades
    // ----------------------------------------------------------------

    /**
     * Retorna una representación textual del libro con sus atributos principales.
     *
     * <p>Formato de ejemplo:</p>
     * <pre>
     * Libro{título='El Quijote', autor='Miguel de Cervantes', ISBN='978-84-376-0494-7', disponible=Sí}
     * </pre>
     *
     * @return cadena con el título, autor, ISBN y estado de disponibilidad del libro
     */
    @Override
    public String toString() {
        return String.format("Libro{título='%s', autor='%s', ISBN='%s', disponible=%s}",
                titulo, autor, isbn, disponible ? "Sí" : "No");
    }

    // ----------------------------------------------------------------
    // Validación interna
    // ----------------------------------------------------------------

    /**
     * Valida que un campo de texto no sea {@code null} ni esté vacío.
     *
     * <p>Método auxiliar extraído durante la fase de refactorización para
     * centralizar la lógica de validación y evitar duplicación de código
     * en el constructor.</p>
     *
     * @param valor        valor del campo a validar
     * @param mensajeError mensaje descriptivo que se incluirá en la excepción
     *                     si la validación falla
     * @throws IllegalArgumentException si {@code valor} es {@code null} o
     *                                  contiene únicamente espacios en blanco
     */
    private void validarCampo(String valor, String mensajeError) {
        if (valor == null || valor.trim().isEmpty()) {
            throw new IllegalArgumentException(mensajeError);
        }
    }
}