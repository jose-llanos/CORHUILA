package com.map.parking_project.Models;

import com.map.parking_project.models.*;
import org.junit.jupiter.api.Test;
import java.time.LocalDate;
import java.time.LocalTime;
import static org.junit.jupiter.api.Assertions.*;

class EntitiesTest {

    @Test
    void testUser() {
        User u = new User();
        u.setId(1L); u.setName("Test"); u.setLastname("User");
        u.setPhone("123"); u.setPlate("ABC"); u.setTypecar("Moto");
        u.setEmail("a@a.com"); u.setPassword("123"); u.setHours(2.5); u.setRol("ADMIN");

        assertEquals(1L, u.getId()); assertEquals("Test", u.getName());
        assertEquals("User", u.getLastname()); assertEquals("123", u.getPhone());
        assertEquals("ABC", u.getPlate()); assertEquals("Moto", u.getTypecar());
        assertEquals("a@a.com", u.getEmail()); assertEquals("123", u.getPassword());
        assertEquals(2.5, u.getHours()); assertEquals("ADMIN", u.getRol());
    }

    @Test
    void testReservas() {
        Reservas r = new Reservas();
        r.setId(1L); r.setTipo_vehiculo("Moto"); r.setTipo_servicio("Parqueo");
        r.setHoras(2); r.setFecha(LocalDate.now()); r.setConfirmada(true); r.setPrecio(10.0);

        assertEquals(1L, r.getId()); assertEquals("Moto", r.getTipo_vehiculo());
        assertEquals("Parqueo", r.getTipo_servicio()); assertEquals(2, r.getHoras());
        assertNotNull(r.getFecha()); assertTrue(r.isConfirmada()); assertEquals(10.0, r.getPrecio());
    }

    @Test
    void testTarifa() {
        Tarifa t = new Tarifa();
        t.setId(1L); t.setTipoVehiculo("Auto"); t.setTarifaDiurna(3000.0);
        t.setTarifaNocturna(4000.0); t.setImagen("url");

        assertEquals(1L, t.getId()); assertEquals("Auto", t.getTipoVehiculo());
        assertEquals(3000.0, t.getTarifaDiurna()); assertEquals(4000.0, t.getTarifaNocturna());
        assertEquals("url", t.getImagen());
    }

    @Test
    void testMapServicesAndVehicleEntry() {
        MapServices ms = new MapServices();
        ms.setId(1L); ms.setName("Z1"); ms.setPrice("100"); ms.setDescription("Desc");
        assertEquals("Z1", ms.getName());

        VehicleEntry ve = new VehicleEntry();
        ve.setId(1L); ve.setPlaca("XYZ"); ve.setTipoVehiculo("Auto");
        ve.setUbicacion("A1"); ve.setHoraIngreso(LocalTime.now()); ve.setFechaIngreso(LocalDate.now());
        assertEquals("XYZ", ve.getPlaca());
    }
}