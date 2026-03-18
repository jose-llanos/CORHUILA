package com.corhuila.calidad;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Stream;

public class UserService {

    private static final Logger logger =
            Logger.getLogger(UserService.class.getName());

    private List<String> users = new ArrayList<>();

    public boolean validateUser(String username) {
        return username != null && !username.isEmpty();
    }

    public String processUser(String user) {
        if (user == null || user.isEmpty()) {
            return "UNKNOWN";
        }
        return classifyUser(user);
    }

    private String classifyUser(String user) {
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

    public void addUser(String username) {
        if (username == null || username.isEmpty()) {
            return;
        }
        users.add(username);
    }

    public int calculateUnusedMetric(int a, int b) {
        return a + b;
    }

    public String readUserFile(String path) throws IOException {
        try (Stream<String> lines =
                     java.nio.file.Files.lines(Paths.get(path))) {
            lines.forEach(logger::info);
        }
        return "File read";
    }

    public List<String> getUsers() {
        return users;
    }
}