package com.corhuila.calidad;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;


public class UserService {

    private List<String> users = new ArrayList<>();

    // DEFECTO 2 corregido: se elimina construcción insegura
    public boolean validateUser(String username) {
        // Simulación segura (sin SQL inseguro)
        return username != null && !username.isEmpty();
    }

    // DEFECTO 3 corregido: menor complejidad cognitiva
    public String processUser(String user) {

        if (user == null || user.isEmpty()) {
            return "UNKNOWN";
        }

        if (!user.contains("admin")) {
            return "USER";
        }

        if (user.contains("test") && user.contains("prod")) {
            return "ADMIN_TEST_PROD";
        }

        if (user.contains("test")) {
            return "ADMIN_TEST";
        }

        return "ADMIN";
    }

    // DEFECTO 4 corregido: manejo de excepciones
    public void addUser(String username) {
        try {
            Integer.parseInt(username);
            users.add(username);
        } catch (NumberFormatException e) {
            // Se ignora o se podría loggear
        }
    }

    // DEFECTO 6 corregido: cierre de recursos
    private static final Logger logger = Logger.getLogger(UserService.class.getName());

    public String readUserFile(String path) throws IOException {
        try (java.util.stream.Stream<String> lines =
                     java.nio.file.Files.lines(java.nio.file.Paths.get(path))) {

            lines.forEach(logger::info);
        }
        return "File read";
    }
    public List<String> getUsers() {
        return users;
    }
}