package com.corhuila.calidad;

/**
 * Clase encargada de realizar conversiones de temperatura entre
 * grados Celsius y Fahrenheit.
 * <p>
 * Soporta valores negativos y precisión decimal.
 * </p>
 *
 * @version 1.0
 */
public class Convertidor {

    /**
     * Convierte una temperatura de grados Celsius a Fahrenheit.
     * <p><b>Requerimiento:</b> RF-001</p>
     *
     * @param c temperatura en grados Celsius
     * @return temperatura convertida a grados Fahrenheit
     */
    public double celsiusAFahrenheit(double c) {
        return (c * 9/5) + 32;
    }

    /**
     * Convierte una temperatura de grados Fahrenheit a Celsius.
     * <p><b>Requerimiento:</b> RF-002</p>
     *
     * @param f temperatura en grados Fahrenheit
     * @return temperatura convertida a grados Celsius
     */
    public double fahrenheitACelsius(double f) {
        return (f - 32) * 5/9;
    }

    /**
     * Verifica si una cadena de texto representa un valor numérico válido.
     * <p><b>Requerimiento:</b> RF-003</p>
     *
     * @param input cadena de texto a validar
     * @return {@code true} si el valor es numérico, {@code false} en caso contrario
     */
    public boolean esNumerico(String input) {
        if (input == null || input.trim().isEmpty()) return false;
        try {
            Double.parseDouble(input);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }
}