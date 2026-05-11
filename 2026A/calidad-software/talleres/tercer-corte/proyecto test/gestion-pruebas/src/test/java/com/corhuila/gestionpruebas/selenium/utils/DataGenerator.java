package com.corhuila.gestionpruebas.selenium.utils;

import java.util.Random;
import java.util.UUID;

public class DataGenerator {

    private static final Random random = new Random();

    public static String generarNombre() {
        String[] nombres = {"Carlos", "Ana", "Luis", "María", "Jorge", "Patricia",
                "Andrés", "Carmen", "Ricardo", "Diana"};
        String[] apellidos = {"García", "Rodríguez", "Martínez", "López", "Sánchez",
                "Pérez", "Gómez", "Díaz", "Fernández", "Torres"};
        return nombres[random.nextInt(nombres.length)] + " " +
                apellidos[random.nextInt(apellidos.length)];
    }

    public static String generarTelefono() {
        return "3" + (100000000 + random.nextInt(900000000));
    }

    public static String generarEmail() {
        return "test_" + UUID.randomUUID().toString().substring(0, 8) + "@test.com";
    }

    public static String generarNombreMascota() {
        String[] nombres = {"Max", "Luna", "Rocky", "Bella", "Coco", "Simba",
                "Nala", "Toby", "Manchas", "Pelusa"};
        return nombres[random.nextInt(nombres.length)];
    }

    public static String generarTexto(int longitud) {
        String caracteres = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ";
        StringBuilder texto = new StringBuilder();
        for (int i = 0; i < longitud; i++) {
            texto.append(caracteres.charAt(random.nextInt(caracteres.length())));
        }
        return texto.toString();
    }
}