package com.autospark.migueljuliana.unitarias.services;
import com.autospark.migueljuliana.models.ReservationRequestDTO;
import org.junit.jupiter.api.Test;
import java.time.LocalDate;
import java.time.LocalTime;
import static org.junit.jupiter.api.Assertions.*;

class ReservationRequestDTOTest {

    @Test
    void testGettersAndSetters() {
        ReservationRequestDTO dto = new ReservationRequestDTO();
        dto.setVehicleType("CARRO");
        dto.setServiceType("Lavado Basico");
        dto.setLicensePlate("ABC123");
        dto.setValue(25000.0);
        dto.setReservationDate(LocalDate.of(2026, 5, 20));
        dto.setReservationTime(LocalTime.of(10, 30));

        assertEquals("CARRO", dto.getVehicleType());
        assertEquals("Lavado Basico", dto.getServiceType());
        assertEquals("ABC123", dto.getLicensePlate());
        assertEquals(25000.0, dto.getValue());
        assertEquals(LocalDate.of(2026, 5, 20), dto.getReservationDate());
        assertEquals(LocalTime.of(10, 30), dto.getReservationTime());
    }
}