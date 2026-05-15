package com.autospark.migueljuliana.unitarias.controllers;

import com.autospark.migueljuliana.controllers.ReservationController;
import com.autospark.migueljuliana.models.Reservation;
import com.autospark.migueljuliana.models.VehicleType;
import com.autospark.migueljuliana.services.IReservationService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.security.oauth2.resource.servlet.OAuth2ResourceServerAutoConfiguration;
import org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration;
import org.springframework.boot.autoconfigure.security.servlet.SecurityFilterAutoConfiguration;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.data.domain.AuditorAware;
import org.springframework.data.jpa.mapping.JpaMetamodelMappingContext;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.List;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(
        controllers = ReservationController.class,
        excludeAutoConfiguration = {
                SecurityAutoConfiguration.class,
                SecurityFilterAutoConfiguration.class,
                OAuth2ResourceServerAutoConfiguration.class
        }
)
@AutoConfigureMockMvc(addFilters = false)
class ReservationControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private IReservationService reservationService;

    @MockitoBean
    private JpaMetamodelMappingContext jpaMetamodelMappingContext;

    @MockitoBean(name = "auditorAwareImpl")
    private AuditorAware<String> auditorAware;

    private Reservation crearReserva() {
        Reservation reservation = new Reservation();
        reservation.setId(1L);
        reservation.setVehicleType(VehicleType.CARRO);
        reservation.setServiceType("Lavado Basico");
        reservation.setLicensePlate("ABC123");
        reservation.setValue(25000.0);
        reservation.setReservationDate(LocalDateTime.of(2026, 5, 20, 10, 30));
        reservation.setActive(true);
        return reservation;
    }

    @Test
    void getAllReservationsDebeRetornarOk() throws Exception {
        when(reservationService.findAll()).thenReturn(List.of(crearReserva()));

        mockMvc.perform(get("/autospark/reserva"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].id").value(1))
                .andExpect(jsonPath("$[0].vehicleType").value("CARRO"))
                .andExpect(jsonPath("$[0].licensePlate").value("ABC123"))
                .andExpect(jsonPath("$[0].serviceType").value("Lavado Basico"));

        verify(reservationService).findAll();
    }

    @Test
    void getReservationByIdDebeRetornarOk() throws Exception {
        when(reservationService.findById(1L)).thenReturn(Optional.of(crearReserva()));

        mockMvc.perform(get("/autospark/reserva/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.vehicleType").value("CARRO"))
                .andExpect(jsonPath("$.licensePlate").value("ABC123"));

        verify(reservationService).findById(1L);
    }

    @Test
    void createReservationDebeRetornarCreated() throws Exception {
        when(reservationService.existsByDateAndTime(LocalDate.of(2026, 5, 20), LocalTime.of(10, 30)))
                .thenReturn(false);
        when(reservationService.save(any(Reservation.class))).thenReturn(crearReserva());

        mockMvc.perform(post("/autospark/reserva")
                        .contentType("application/json")
                        .content("""
                                {
                                  "vehicleType": "CARRO",
                                  "serviceType": "Lavado Basico",
                                  "licensePlate": "ABC123",
                                  "value": 25000.0,
                                  "reservationDate": "2026-05-20",
                                  "reservationTime": "10:30:00"
                                }
                                """))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.vehicleType").value("CARRO"))
                .andExpect(jsonPath("$.licensePlate").value("ABC123"))
                .andExpect(jsonPath("$.active").value(true));

        verify(reservationService).existsByDateAndTime(LocalDate.of(2026, 5, 20), LocalTime.of(10, 30));
        verify(reservationService).save(any(Reservation.class));
    }

    @Test
    void createReservationConHorarioOcupadoDebeRetornarConflict() throws Exception {
        when(reservationService.existsByDateAndTime(LocalDate.of(2026, 5, 20), LocalTime.of(10, 30)))
                .thenReturn(true);

        mockMvc.perform(post("/autospark/reserva")
                        .contentType("application/json")
                        .content("""
                                {
                                  "vehicleType": "CARRO",
                                  "serviceType": "Lavado Basico",
                                  "licensePlate": "ABC123",
                                  "value": 25000.0,
                                  "reservationDate": "2026-05-20",
                                  "reservationTime": "10:30:00"
                                }
                                """))
                .andExpect(status().isConflict())
                .andExpect(content().string("Ya existe una reserva en esta fecha y hora. Por favor selecciona otro horario."));

        verify(reservationService, never()).save(any(Reservation.class));
    }

    @Test
    void updateReservationDebeRetornarOk() throws Exception {
        Reservation existing = crearReserva();

        Reservation updated = crearReserva();
        updated.setServiceType("Lavado Premium");
        updated.setValue(50000.0);

        when(reservationService.findById(1L)).thenReturn(Optional.of(existing));
        when(reservationService.save(any(Reservation.class))).thenReturn(updated);

        mockMvc.perform(put("/autospark/reserva/1")
                        .contentType("application/json")
                        .content("""
                                {
                                  "vehicleType": "CARRO",
                                  "serviceType": "Lavado Premium",
                                  "licensePlate": "ABC123",
                                  "value": 50000.0,
                                  "reservationDate": "2026-05-20T10:30:00",
                                  "active": true
                                }
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.serviceType").value("Lavado Premium"))
                .andExpect(jsonPath("$.value").value(50000.0));

        verify(reservationService).findById(1L);
        verify(reservationService).save(any(Reservation.class));
    }

    @Test
    void deleteReservationDebeRetornarNoContent() throws Exception {
        when(reservationService.findById(1L)).thenReturn(Optional.of(crearReserva()));
        doNothing().when(reservationService).delete(1L);

        mockMvc.perform(delete("/autospark/reserva/1"))
                .andExpect(status().isNoContent());

        verify(reservationService).findById(1L);
        verify(reservationService).delete(1L);
    }

    @Test
    void getReservationsWithUsersDebeRetornarOk() throws Exception {
        when(reservationService.getReservationsWithUsers()).thenReturn(List.of());

        mockMvc.perform(get("/autospark/reservas-con-usuarios"))
                .andExpect(status().isOk());

        verify(reservationService).getReservationsWithUsers();
    }

    @Test
    void activateReservationDebeRetornarOk() throws Exception {
        Reservation reservation = crearReserva();
        reservation.setActive(true);

        when(reservationService.activateReservation(1L)).thenReturn(Optional.of(reservation));

        mockMvc.perform(put("/autospark/reserva/1/activar"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.active").value(true));

        verify(reservationService).activateReservation(1L);
    }

    @Test
    void deactivateReservationDebeRetornarOk() throws Exception {
        Reservation reservation = crearReserva();
        reservation.setActive(false);

        when(reservationService.deactivateReservation(1L)).thenReturn(Optional.of(reservation));

        mockMvc.perform(put("/autospark/reserva/1/desactivar"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.active").value(false));

        verify(reservationService).deactivateReservation(1L);
    }
}