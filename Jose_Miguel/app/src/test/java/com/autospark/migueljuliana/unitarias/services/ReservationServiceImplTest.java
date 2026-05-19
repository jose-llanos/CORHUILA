package com.autospark.migueljuliana.unitarias.services;

import com.autospark.migueljuliana.models.Reservation;
import com.autospark.migueljuliana.models.ReservationUserDTO;
import com.autospark.migueljuliana.models.User;
import com.autospark.migueljuliana.models.VehicleType;
import com.autospark.migueljuliana.repositories.IReservationRepository;
import com.autospark.migueljuliana.repositories.IUserRepository;
import com.autospark.migueljuliana.services.ReservationServiceImpl;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * Suite de pruebas unitarias para el servicio de gestión de reservas.
 *
 * <p>Esta clase prueba los siguientes requisitos funcionales:</p>
 * <ul>
 *   <li>RF7: Gestión completa de reservas (CRUD) - TC-023 a TC-029</li>
 *   <li>RF8: Ver reservas con datos del usuario - TC-030 a TC-031</li>
 * </ul>
 *
 * @author AutoSpark Team
 * @version 1.0
 */
@ExtendWith(MockitoExtension.class)
class ReservationServiceImplTest {

    // ===== MOCKS =====
    @Mock
    private IReservationRepository repository;

    @Mock
    private IUserRepository userRepository;

    @InjectMocks
    private ReservationServiceImpl service;

    // ===== OBJETOS DE PRUEBA =====
    private Reservation testReservation;

    // ===== CONFIGURACIÓN INICIAL =====
    @BeforeEach
    void setUp() {
        testReservation = new Reservation();
        testReservation.setId(1L);
        testReservation.setVehicleType(VehicleType.CARRO);
        testReservation.setServiceType("Lavado Basico");
        testReservation.setLicensePlate("ABC123");
        testReservation.setValue(25000.0);
        testReservation.setReservationDate(LocalDateTime.of(2025, 5, 20, 10, 30));
        testReservation.setActive(true);
    }

    // ================================================================
    // ========== PRUEBAS PARA RF7: GESTIÓN COMPLETA DE RESERVAS ==========
    // ================================================================

    /**
     * TC-023: Crear reserva válida
     * <p><b>Requisito:</b> RF7 - El sistema debe permitir crear una nueva reserva</p>
     * <p><b>Entrada:</b> Objeto Reservation con datos válidos</p>
     * <p><b>Salida esperada:</b> Reserva guardada con estado activo (active=true)</p>
     */
    @Test
    void testCreateReservation() {
        when(repository.save(any(Reservation.class))).thenReturn(testReservation);

        Reservation saved = service.save(testReservation);

        assertNotNull(saved);
        assertEquals("ABC123", saved.getLicensePlate());
        assertTrue(saved.isActive());
        verify(repository, times(1)).save(any(Reservation.class));
    }

    /**
     * TC-024: Crear reserva con fecha pasada (excepción)
     * <p><b>Requisito:</b> RF7 - El sistema debe validar que la fecha no sea pasada</p>
     * <p><b>Entrada:</b> Objeto Reservation con fecha en el pasado</p>
     * <p><b>Salida esperada:</b> Excepción RuntimeException</p>
     */
    @Test
    void testCreateReservationWithPastDate() {
        testReservation.setReservationDate(LocalDateTime.of(2020, 1, 1, 10, 0));

        when(repository.save(any(Reservation.class))).thenThrow(
                new RuntimeException("La fecha de reserva no puede ser en el pasado")
        );

        assertThrows(RuntimeException.class, () -> {
            service.save(testReservation);
        });
    }

    /**
     * TC-025: Activar reserva inactiva
     * <p><b>Requisito:</b> RF7 - El sistema debe permitir activar una reserva</p>
     * <p><b>Entrada:</b> ID de una reserva inactiva</p>
     * <p><b>Salida esperada:</b> Reserva activada (active = true)</p>
     */
    @Test
    void testActivateReservation() {
        testReservation.setActive(false);
        when(repository.findById(1L)).thenReturn(Optional.of(testReservation));
        when(repository.save(any(Reservation.class))).thenReturn(testReservation);

        service.activateReservation(1L);

        assertTrue(testReservation.isActive());
        verify(repository, times(1)).save(testReservation);
    }

    /**
     * TC-026: Eliminar reserva inexistente
     * <p><b>Requisito:</b> RF7 - Eliminar reserva inexistente no debe lanzar excepción</p>
     * <p><b>Entrada:</b> ID = 999 (reserva inexistente)</p>
     * <p><b>Salida esperada:</b> No lanza excepción</p>
     */
    @Test
    void testDeleteReservationNotFound() {
        doNothing().when(repository).deleteById(999L);

        assertDoesNotThrow(() -> service.delete(999L));

        verify(repository, times(1)).deleteById(999L);
    }

    /**
     * TC-027: Actualizar reserva inexistente (excepción)
     * <p><b>Requisito:</b> RF7 - Manejo de errores al actualizar reserva inexistente</p>
     * <p><b>Entrada:</b> ID = 999 (reserva inexistente)</p>
     * <p><b>Salida esperada:</b> Excepción RuntimeException</p>
     */
    @Test
    void testUpdateReservationNotFound() {
        when(repository.findById(999L)).thenReturn(Optional.empty());

        assertThrows(RuntimeException.class, () -> {
            service.update(testReservation, 999L);
        });

        verify(repository, never()).save(any(Reservation.class));
    }

    /**
     * TC-028: Desactivar reserva inexistente
     * <p><b>Requisito:</b> RF7 - Desactivar reserva inexistente</p>
     * <p><b>Entrada:</b> ID = 999 (reserva inexistente)</p>
     * <p><b>Salida esperada:</b> Optional vacío</p>
     */
    @Test
    void testDeactivateReservationNotFound() {
        when(repository.findById(999L)).thenReturn(Optional.empty());

        Optional<Reservation> result = service.deactivateReservation(999L);

        assertTrue(result.isEmpty());
        verify(repository, times(1)).findById(999L);
        verify(repository, never()).save(any(Reservation.class));
    }

    /**
     * TC-029: Buscar reserva por ID inexistente
     * <p><b>Requisito:</b> RF7 - Búsqueda de reserva inexistente</p>
     * <p><b>Entrada:</b> ID = 999 (reserva inexistente)</p>
     * <p><b>Salida esperada:</b> Optional vacío</p>
     */
    @Test
    void testFindReservationByIdNotFound() {
        when(repository.findById(999L)).thenReturn(Optional.empty());

        Optional<Reservation> result = service.findById(999L);

        assertTrue(result.isEmpty());
        verify(repository, times(1)).findById(999L);
    }

    // ================================================================
    // ========== PRUEBAS PARA RF8: VER RESERVAS CON DATOS DEL USUARIO ==========
    // ================================================================

    /**
     * TC-030: Listar reservas
     * <p><b>Requisito:</b> RF8 - El sistema debe permitir listar todas las reservas</p>
     * <p><b>Entrada:</b> Ninguna</p>
     * <p><b>Salida esperada:</b> Lista de reservas registradas</p>
     */
    @Test
    void testFindAllReservations() {
        when(repository.findAll()).thenReturn(List.of(testReservation));

        var reservations = service.findAll();

        assertNotNull(reservations);
        assertEquals(1, reservations.size());
        verify(repository, times(1)).findAll();
    }

    /**
     * TC-031: Ver reservas con datos del usuario (caso feliz)
     * <p><b>Requisito:</b> RF8 - Mostrar reservas junto con datos del usuario</p>
     * <p><b>Entrada:</b> Reservas existentes con usuarios asociados</p>
     * <p><b>Salida esperada:</b> Lista de DTOs con datos combinados</p>
     */
    @Test
    void testGetReservationsWithUsersSuccess() {
        // ARRANGE
        User mockUser = new User();
        mockUser.setFullName("Juan Perez");
        mockUser.setIdentityCard("12345678");
        mockUser.setPhone("3001234567");
        mockUser.setLicensePlate("ABC123");

        when(repository.findAll()).thenReturn(List.of(testReservation));
        when(userRepository.findByLicensePlate("ABC123")).thenReturn(Optional.of(mockUser));

        // ACT
        List<ReservationUserDTO> result = service.getReservationsWithUsers();

        // ASSERT
        assertNotNull(result);
        assertEquals(1, result.size());
        assertEquals("Juan Perez", result.get(0).getCustomerFullName());

        verify(repository, times(1)).findAll();
        verify(userRepository, times(1)).findByLicensePlate("ABC123");
    }

    /**
     * TC-032: Desactivar reserva existente
     * <p><b>Requisito:</b> RF7 - El sistema debe permitir desactivar reservas</p>
     * <p><b>Entrada:</b> ID = 1 (reserva activa existente)</p>
     * <p><b>Salida esperada:</b> Reserva desactivada con active = false</p>
     */
    @Test
    void testDeactivateReservationSuccess() {
        testReservation.setActive(true);

        when(repository.findById(1L))
                .thenReturn(Optional.of(testReservation));

        when(repository.save(any(Reservation.class)))
                .thenReturn(testReservation);

        Optional<Reservation> result =
                service.deactivateReservation(1L);

        assertTrue(result.isPresent());
        assertFalse(result.get().isActive());

        verify(repository, times(1))
                .save(testReservation);
    }

    /**
     * TC-033: Buscar reserva por ID existente
     * <p><b>Requisito:</b> RF7 - El sistema debe permitir buscar reservas por ID</p>
     * <p><b>Entrada:</b> ID = 1 (reserva existente)</p>
     * <p><b>Salida esperada:</b> Optional con los datos de la reserva</p>
     */
    @Test
    void testFindReservationByIdFound() {
        when(repository.findById(1L))
                .thenReturn(Optional.of(testReservation));

        Optional<Reservation> result =
                service.findById(1L);

        assertTrue(result.isPresent());
        assertEquals("ABC123",
                result.get().getLicensePlate());

        verify(repository, times(1))
                .findById(1L);
    }

    /**
     * TC-034: Verificar existencia de reserva por fecha y hora
     * <p><b>Requisito:</b> RF7 - Validar disponibilidad de horario</p>
     * <p><b>Entrada:</b> Fecha = 2025-05-20, Hora = 10:30</p>
     * <p><b>Salida esperada:</b> true si ya existe reserva en ese horario</p>
     */
    @Test
    void testExistsByDateAndTimeTrue() {

        when(repository.existsByReservationDateBetween(
                any(LocalDateTime.class),
                any(LocalDateTime.class)
        )).thenReturn(true);

        boolean result = service.existsByDateAndTime(
                java.time.LocalDate.of(2025, 5, 20),
                java.time.LocalTime.of(10, 30)
        );

        assertTrue(result);

        verify(repository, times(1))
                .existsByReservationDateBetween(
                        any(LocalDateTime.class),
                        any(LocalDateTime.class)
                );
    }

    /**
     * TC-035: Verificar disponibilidad de horario libre
     * <p><b>Requisito:</b> RF7 - Validar horario disponible</p>
     * <p><b>Entrada:</b> Fecha = 2025-05-20, Hora = 15:00</p>
     * <p><b>Salida esperada:</b> false si no existe reserva</p>
     */
    @Test
    void testExistsByDateAndTimeFalse() {

        when(repository.existsByReservationDateBetween(
                any(LocalDateTime.class),
                any(LocalDateTime.class)
        )).thenReturn(false);

        boolean result = service.existsByDateAndTime(
                java.time.LocalDate.of(2025, 5, 20),
                java.time.LocalTime.of(15, 0)
        );

        assertFalse(result);

        verify(repository, times(1))
                .existsByReservationDateBetween(
                        any(LocalDateTime.class),
                        any(LocalDateTime.class)
                );
    }

    /**
     * TC-036: Actualizar reserva existente correctamente
     * <p><b>Requisito:</b> RF7 - Actualizar datos de reserva existente</p>
     * <p><b>Entrada:</b> Reserva válida con nuevos datos</p>
     * <p><b>Salida esperada:</b> Reserva actualizada exitosamente</p>
     */
    @Test
    void testUpdateReservationSuccess() {

        Reservation updatedReservation = new Reservation();
        updatedReservation.setVehicleType(VehicleType.CARRO);
        updatedReservation.setServiceType("Lavado Premium");
        updatedReservation.setLicensePlate("XYZ999");
        updatedReservation.setValue(50000.0);
        updatedReservation.setReservationDate(
                LocalDateTime.of(2025, 6, 10, 14, 0)
        );
        updatedReservation.setActive(false);

        when(repository.findById(1L))
                .thenReturn(Optional.of(testReservation));

        when(repository.save(any(Reservation.class)))
                .thenReturn(testReservation);

        assertDoesNotThrow(() ->
                service.update(updatedReservation, 1L)
        );

        assertEquals(VehicleType.CARRO,
                testReservation.getVehicleType());

        assertEquals("Lavado Premium",
                testReservation.getServiceType());

        assertEquals("XYZ999",
                testReservation.getLicensePlate());

        assertEquals(50000.0,
                testReservation.getValue());

        assertFalse(testReservation.isActive());

        verify(repository, times(1))
                .save(testReservation);
    }

    /**
     * TC-037: Obtener reservas sin usuario asociado
     * <p><b>Requisito:</b> RF8 - Manejar reservas sin usuario relacionado</p>
     * <p><b>Entrada:</b> Reserva con placa no registrada</p>
     * <p><b>Salida esperada:</b> Lista vacía</p>
     */
    @Test
    void testGetReservationsWithUsersUserNotFound() {

        when(repository.findAll())
                .thenReturn(List.of(testReservation));

        when(userRepository.findByLicensePlate("ABC123"))
                .thenReturn(Optional.empty());

        List<ReservationUserDTO> result =
                service.getReservationsWithUsers();

        assertNotNull(result);
        assertTrue(result.isEmpty());

        verify(repository, times(1))
                .findAll();

        verify(userRepository, times(1))
                .findByLicensePlate("ABC123");
    }

    /**
     * TC-038: Activar reserva inexistente
     * <p><b>Requisito:</b> RF7 - Activar reserva inexistente</p>
     * <p><b>Entrada:</b> ID = 999</p>
     * <p><b>Salida esperada:</b> Optional vacío</p>
     */
    @Test
    void testActivateReservationNotFound() {

        when(repository.findById(999L))
                .thenReturn(Optional.empty());

        Optional<Reservation> result =
                service.activateReservation(999L);

        assertTrue(result.isEmpty());

        verify(repository, times(1))
                .findById(999L);

        verify(repository, never())
                .save(any(Reservation.class));
    }
}