package com.map.parking.selenium.support;

import com.map.parking.selenium.config.SeleniumConfig;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;

/**
 * Crea el usuario administrador de prueba vía API REST (no requiere usar el front manualmente).
 */
public final class TestUserSetup {

    private TestUserSetup() {
    }

    public static void ensureAdminUserExists() {
        try {
            if (userExists(SeleniumConfig.TEST_ADMIN_EMAIL)) {
                return;
            }
            createAdminUser();
        } catch (Exception e) {
            throw new IllegalStateException(
                    "No se pudo preparar el usuario de prueba. Verifica Docker (API en :8080).", e);
        }
    }

    private static boolean userExists(String email) throws Exception {
        HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(SeleniumConfig.API_URL + "/api/user"))
                .GET()
                .build();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        return response.statusCode() == 200 && response.body().contains(email);
    }

    private static void createAdminUser() throws Exception {
        String json = """
                {
                  "name": "Selenium",
                  "lastname": "Admin",
                  "phone": "3001234567",
                  "plate": "SEL9999",
                  "typecar": "Automóvil",
                  "email": "%s",
                  "password": "%s",
                  "rol": "Administrador",
                  "hours": 0
                }
                """.formatted(SeleniumConfig.TEST_ADMIN_EMAIL, SeleniumConfig.TEST_ADMIN_PASSWORD);

        HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(SeleniumConfig.API_URL + "/api/user"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json, StandardCharsets.UTF_8))
                .build();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 201 && response.statusCode() != 200) {
            throw new IllegalStateException("POST /api/user falló: " + response.statusCode() + " " + response.body());
        }
    }
}
