package com.corhuila.unitest;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Gestiona una lista de estudiantes con operaciones CRUD y consultas.
 */
public class GestorEstudiantes {

    private final List<Estudiante> estudiantes = new ArrayList<>();

    /**
     * Agrega un estudiante. Lanza excepción si ya existe el ID.
     */
    public void agregar(Estudiante estudiante) {
        if (estudiante == null)
            throw new IllegalArgumentException("El estudiante no puede ser nulo.");
        boolean existe = estudiantes.stream()
                .anyMatch(e -> e.getId().equals(estudiante.getId()));
        if (existe)
            throw new IllegalStateException("Ya existe un estudiante con el ID: " + estudiante.getId());
        estudiantes.add(estudiante);
    }

    /**
     * Elimina un estudiante por ID. Lanza excepción si no existe.
     */
    public void eliminar(String id) {
        Estudiante encontrado = buscarPorId(id)
                .orElseThrow(() -> new IllegalArgumentException("No existe estudiante con ID: " + id));
        estudiantes.remove(encontrado);
    }

    /**
     * Busca un estudiante por ID.
     */
    public Optional<Estudiante> buscarPorId(String id) {
        return estudiantes.stream()
                .filter(e -> e.getId().equals(id))
                .findFirst();
    }

    /**
     * Actualiza la nota de un estudiante.
     */
    public void actualizarNota(String id, double nuevaNota) {
        Estudiante estudiante = buscarPorId(id)
                .orElseThrow(() -> new IllegalArgumentException("No existe estudiante con ID: " + id));
        estudiante.setNota(nuevaNota);
    }

    /**
     * Retorna la cantidad total de estudiantes registrados.
     */
    public int contarEstudiantes() {
        return estudiantes.size();
    }

    /**
     * Retorna el promedio de notas de todos los estudiantes.
     * Lanza excepción si no hay estudiantes.
     */
    public double calcularPromedio() {
        if (estudiantes.isEmpty())
            throw new IllegalStateException("No hay estudiantes registrados para calcular el promedio.");
        return estudiantes.stream()
                .mapToDouble(Estudiante::getNota)
                .average()
                .orElse(0.0);
    }

    /**
     * Retorna la lista de estudiantes que aprobaron (nota >= 3.0).
     */
    public List<Estudiante> obtenerAprobados() {
        return estudiantes.stream()
                .filter(Estudiante::aprobado)
                .collect(Collectors.toList());
    }

    /**
     * Retorna el estudiante con la nota más alta.
     * Lanza excepción si no hay estudiantes.
     */
    public Estudiante obtenerMejorEstudiante() {
        if (estudiantes.isEmpty())
            throw new IllegalStateException("No hay estudiantes registrados.");
        return estudiantes.stream()
                .max(Comparator.comparingDouble(Estudiante::getNota))
                .orElseThrow();
    }

    /**
     * Retorna todos los estudiantes registrados (copia defensiva).
     */
    public List<Estudiante> listarTodos() {
        return new ArrayList<>(estudiantes);
    }
}