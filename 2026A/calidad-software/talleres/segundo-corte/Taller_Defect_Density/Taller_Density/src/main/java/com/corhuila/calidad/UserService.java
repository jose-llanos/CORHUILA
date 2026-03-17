package com.corhuila.calidad;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Stream;

public class UserService {
    private static final Logger LOGGER = Logger.getLogger(UserService.class.getName());

    private final List<String> users = new ArrayList<>();

    public boolean validateUser(String username) {
        if (username == null) {
            return false;
        }

        String normalizedUser = username.trim();
        return !normalizedUser.isEmpty();
    }

    public String processUser(String user) {
        if (user == null || user.isEmpty()) {
            return "UNKNOWN";
        }

        boolean containsAdmin = user.contains("admin");
        boolean containsTest = user.contains("test");
        boolean containsProd = user.contains("prod");

        if (!containsAdmin) {
            return "USER";
        }

        if (containsTest && containsProd) {
            return "ADMIN_TEST_PROD";
        }

        if (containsTest) {
            return "ADMIN_TEST";
        }

        return "ADMIN";
    }

    public void addUser(String username) {
        if (!validateUser(username)) {
            LOGGER.warning("No se puede agregar un usuario vacío o nulo.");
            return;
        }

        users.add(username.trim());
    }

    public String readUserFile(String path) throws IOException {
        try (Stream<String> lines = Files.lines(Path.of(path))) {
            lines.forEach(LOGGER::info);
        }
        return "File read";
    }

    public List<String> getUsers() {
        return new ArrayList<>(users);
    }
}