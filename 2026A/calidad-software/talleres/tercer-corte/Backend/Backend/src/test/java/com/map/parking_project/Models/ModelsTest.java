package com.map.parking_project.Models;

import com.map.parking_project.models.Tarifa;
import com.map.parking_project.models.User;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ModelsTest {
    @Test
    void testUserEntity() {
        User user = new User();
        user.setId(1L);
        user.setEmail("test@mail.com");
        user.setPassword("hash123");
        assertEquals(1L, user.getId());
        assertEquals("test@mail.com", user.getEmail());
    }

    @Test
    void testTarifaEntity() {
        Tarifa t = new Tarifa();
        t.setId(10L);
        // Agrega aquí los setters de tu clase Tarifa
        assertEquals(10L, t.getId());
    }

    // Haz lo mismo para VehicleEntry y Reserva. ¡Esto suma mucho porcentaje!
}