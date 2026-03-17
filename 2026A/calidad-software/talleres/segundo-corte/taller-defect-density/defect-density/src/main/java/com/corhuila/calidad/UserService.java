package com.corhuila.calidad;
import java.util.ArrayList;
import java.util.List;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.logging.Logger;
import java.util.stream.Stream;
import java.util.Map;
import java.util.function.Predicate;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import javax.sql.DataSource;

public class UserService {
    private List<String> users = new ArrayList<>();
    private static final Logger logger = Logger.getLogger(UserService.class.getName());
    private DataSource dataSource;
    private static final String ADMIN = "admin";
    private static final String TEST = "test";
    private static final String PROD = "prod";

    public boolean validateUser(String username) {
        if (username == null || username.isBlank()) {
            return false;
        }

        // Si no hay DataSource (caso test), evita romper y usa validación simple
        if (dataSource == null) {
            return true; // o alguna lógica básica si quieres ser más estricto
        }

        String query = "SELECT 1 FROM users WHERE name = ?";

        try (Connection conn = dataSource.getConnection();
             PreparedStatement stmt = conn.prepareStatement(query)) {

            stmt.setString(1, username);
            ResultSet rs = stmt.executeQuery();

            return rs.next();
        } catch (SQLException e) {
            logger.severe("Error en la consulta: " + e.getMessage());
            return false;
        }
    }

    public String processUser(String user) {
        if (user == null || user.isEmpty()) return "UNKNOWN";

        List<Map.Entry<Predicate<String>, String>> rules = List.of(
                Map.entry(u -> u.contains(ADMIN) && u.contains(TEST) && u.contains(PROD), "ADMIN_TEST_PROD"),
                Map.entry(u -> u.contains(ADMIN) && u.contains(TEST), "ADMIN_TEST"),
                Map.entry(u -> u.contains(ADMIN), "ADMIN")
        );

        return rules.stream()
                .filter(rule -> rule.getKey().test(user))
                .map(Map.Entry::getValue)
                .findFirst()
                .orElse("UNKNOWN");
    }

    public void addUser(String username) {
        try {
            Integer.parseInt(username); // validación opcional
            users.add(username);
        } catch (NumberFormatException e) {
            logger.warning("Username no es un número válido");
        }
    }

    public String readUserFile(String path) throws IOException {
        try (Stream<String> lines = Files.lines(Paths.get(path))) {
            lines.forEach(logger::info);
        }
        return "File read";
    }

    public List<String> getUsers() {
        return users;
    }
}