package com.sgplab.backend;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

/**
 * Verifica que el contexto de Spring se carga correctamente
 * con el perfil de pruebas.
 */
@SpringBootTest
@ActiveProfiles("test")
class SgpBackendApplicationTests {

    @Test
    void contextLoads() {
        // Si el contexto no carga, este test falla con detalles del bean fallido.
    }
}
