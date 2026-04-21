package com.corhuila.unitest;

/**
 * Representa un estudiante con sus datos básicos y nota.
 */
public class Estudiante {

    private final String id;
    private final String nombre;
    private final int edad;
    private double nota; // 0.0 a 5.0

    public Estudiante(String id, String nombre, int edad, double nota) {
        if (id == null || id.isBlank())
            throw new IllegalArgumentException("El ID no puede ser nulo o vacío.");
        if (nombre == null || nombre.isBlank())
            throw new IllegalArgumentException("El nombre no puede ser nulo o vacío.");
        if (edad < 5 || edad > 100)
            throw new IllegalArgumentException("La edad debe estar entre 5 y 100 años.");
        if (nota < 0.0 || nota > 5.0)
            throw new IllegalArgumentException("La nota debe estar entre 0.0 y 5.0.");

        this.id = id;
        this.nombre = nombre;
        this.edad = edad;
        this.nota = nota;
    }

    public String getId()      { return id; }
    public String getNombre()  { return nombre; }
    public int getEdad()       { return edad; }
    public double getNota()    { return nota; }

    public void setNota(double nota) {
        if (nota < 0.0 || nota > 5.0)
            throw new IllegalArgumentException("La nota debe estar entre 0.0 y 5.0.");
        this.nota = nota;
    }

    public boolean aprobado() {
        return this.nota >= 3.0;
    }

    @Override
    public String toString() {
        return String.format("Estudiante{id='%s', nombre='%s', edad=%d, nota=%.1f}",
                id, nombre, edad, nota);
    }
}