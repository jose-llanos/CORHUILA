package com.medicita.app.service.impl;

import com.medicita.app.dto.schedule.DoctorAvailabilityDTO;
import com.medicita.app.dto.schedule.DoctorScheduleDTO;
import com.medicita.app.dto.schedule.DoctorScheduleRequest;
import com.medicita.app.entity.Appointment;
import com.medicita.app.entity.Doctor;
import com.medicita.app.entity.DoctorLeave;
import com.medicita.app.entity.DoctorSchedule;
import com.medicita.app.entity.User;
import com.medicita.app.enums.AppointmentStatus;
import com.medicita.app.enums.LeaveStatus;
import com.medicita.app.enums.LeaveType;
import com.medicita.app.enums.Role;
import com.medicita.app.enums.Weekday;
import com.medicita.app.exception.ResourceNotFoundException;
import com.medicita.app.repository.AppointmentRepository;
import com.medicita.app.repository.DoctorLeaveRepository;
import com.medicita.app.repository.DoctorRepository;
import com.medicita.app.repository.DoctorScheduleRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/*
 * Pruebas unitarias para DoctorScheduleServiceImpl.
 * Este servicio es el más "técnico" del sistema: maneja los horarios semanales
 * de los médicos y el cálculo de disponibilidad hora a hora.
 * Lo interesante está en getAvailability(), que cruza tres fuentes de datos:
 * el horario del médico, sus permisos aprobados y las citas ya agendadas.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("DoctorScheduleServiceImpl — Pruebas unitarias")
class DoctorScheduleServiceImplTest {

    @Mock private DoctorScheduleRepository doctorScheduleRepository;
    @Mock private DoctorRepository doctorRepository;
    @Mock private AppointmentRepository appointmentRepository;
    @Mock private DoctorLeaveRepository doctorLeaveRepository;

    @InjectMocks
    private DoctorScheduleServiceImpl doctorScheduleService;

    private User doctorUser;
    private Doctor doctor;
    private DoctorSchedule mondaySchedule;
    private DoctorScheduleRequest mondayRequest;

    // Usamos lunes 2027-03-08 para que date.getDayOfWeek().name() == "MONDAY"
    private static final LocalDate MONDAY = LocalDate.of(2027, 3, 8);

    @BeforeEach
    void setUp() {
        doctorUser = User.builder()
                .id(UUID.randomUUID())
                .firstName("Ana")
                .lastName("Gómez")
                .email("ana@medicita.com")
                .role(Role.DOCTOR)
                .active(true)
                .build();

        doctor = Doctor.builder()
                .id(UUID.randomUUID())
                .user(doctorUser)
                .medicalLicense("MED-030")
                .active(true)
                .build();

        mondaySchedule = DoctorSchedule.builder()
                .id(UUID.randomUUID())
                .doctor(doctor)
                .weekDay(Weekday.MONDAY)
                .startTime(LocalTime.of(8, 0))
                .endTime(LocalTime.of(12, 0))
                .active(true)
                .build();

        mondayRequest = DoctorScheduleRequest.builder()
                .weekDay(Weekday.MONDAY)
                .startTime(LocalTime.of(8, 0))
                .endTime(LocalTime.of(12, 0))
                .active(true)
                .build();
    }

    // =========================================================================
    // findByDoctor()
    // =========================================================================

    @Test
    @DisplayName("findByDoctor: devuelve la lista de horarios del médico")
    void findByDoctor_doctorExiste_devuelveHorarios() {
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        when(doctorScheduleRepository.findByDoctor(doctor)).thenReturn(List.of(mondaySchedule));

        List<DoctorScheduleDTO> result = doctorScheduleService.findByDoctor(doctor.getId());

        assertThat(result).hasSize(1);
        assertThat(result.get(0).getDayOfWeek()).isEqualTo("MONDAY");
    }

    @Test
    @DisplayName("findByDoctor: lanza ResourceNotFoundException si el médico no existe")
    void findByDoctor_doctorNoExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(doctorRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> doctorScheduleService.findByDoctor(fakeId))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    // =========================================================================
    // create()
    // =========================================================================

    @Test
    @DisplayName("create: guarda el horario y devuelve el DTO correspondiente")
    void create_conDatosValidos_guardaYDevuelveDTO() {
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        when(doctorScheduleRepository.save(any(DoctorSchedule.class))).thenReturn(mondaySchedule);

        DoctorScheduleDTO result = doctorScheduleService.create(doctor.getId(), mondayRequest);

        assertThat(result).isNotNull();
        assertThat(result.getDayOfWeek()).isEqualTo("MONDAY");
        assertThat(result.getStartTime()).isEqualTo(LocalTime.of(8, 0));
        verify(doctorScheduleRepository).save(any(DoctorSchedule.class));
    }

    @Test
    @DisplayName("create: lanza ResourceNotFoundException si el médico no existe")
    void create_doctorNoExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(doctorRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> doctorScheduleService.create(fakeId, mondayRequest))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    // =========================================================================
    // update()
    // =========================================================================

    @Test
    @DisplayName("update: cambia el día y las horas del horario existente")
    void update_horarioExiste_actualizaDatos() {
        DoctorScheduleRequest updReq = DoctorScheduleRequest.builder()
                .weekDay(Weekday.TUESDAY)
                .startTime(LocalTime.of(14, 0))
                .endTime(LocalTime.of(18, 0))
                .build();

        when(doctorScheduleRepository.findById(mondaySchedule.getId()))
                .thenReturn(Optional.of(mondaySchedule));
        when(doctorScheduleRepository.save(any())).thenReturn(mondaySchedule);

        DoctorScheduleDTO result = doctorScheduleService.update(mondaySchedule.getId(), updReq);

        // La entidad fue mutada in-place antes del save
        assertThat(mondaySchedule.getWeekDay()).isEqualTo(Weekday.TUESDAY);
        assertThat(mondaySchedule.getStartTime()).isEqualTo(LocalTime.of(14, 0));
        assertThat(result).isNotNull();
        verify(doctorScheduleRepository).save(mondaySchedule);
    }

    @Test
    @DisplayName("update: lanza ResourceNotFoundException si el horario no existe")
    void update_horarioNoExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(doctorScheduleRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> doctorScheduleService.update(fakeId, mondayRequest))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    // =========================================================================
    // delete()
    // =========================================================================

    @Test
    @DisplayName("delete: hace soft-delete poniendo active=false")
    void delete_horarioExiste_loDesactiva() {
        when(doctorScheduleRepository.findById(mondaySchedule.getId()))
                .thenReturn(Optional.of(mondaySchedule));

        doctorScheduleService.delete(mondaySchedule.getId());

        assertThat(mondaySchedule.isActive()).isFalse();
        verify(doctorScheduleRepository).save(mondaySchedule);
    }

    @Test
    @DisplayName("delete: lanza ResourceNotFoundException si el horario no existe")
    void delete_horarioNoExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(doctorScheduleRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> doctorScheduleService.delete(fakeId))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    // =========================================================================
    // replaceWeekly()
    // =========================================================================

    @Test
    @DisplayName("replaceWeekly: crea una nueva entrada cuando el día no existía antes")
    void replaceWeekly_diaNoExistia_creaEntradaNueva() {
        // No hay horarios previos del médico
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        when(doctorScheduleRepository.findByDoctor(doctor))
                .thenReturn(List.of())          // primera llamada: vacío
                .thenReturn(List.of(mondaySchedule)); // segunda llamada: resultado final

        List<DoctorScheduleDTO> result = doctorScheduleService.replaceWeekly(
                doctor.getId(), List.of(mondayRequest));

        assertThat(result).hasSize(1);
        // Se guardó la nueva entrada
        verify(doctorScheduleRepository).save(any(DoctorSchedule.class));
    }

    @Test
    @DisplayName("replaceWeekly: actualiza la entrada existente si el día ya estaba configurado")
    void replaceWeekly_diaExistia_actualizaEntrada() {
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        when(doctorScheduleRepository.findByDoctor(doctor))
                .thenReturn(List.of(mondaySchedule))
                .thenReturn(List.of(mondaySchedule));

        DoctorScheduleRequest updatedMonday = DoctorScheduleRequest.builder()
                .weekDay(Weekday.MONDAY)
                .startTime(LocalTime.of(9, 0))
                .endTime(LocalTime.of(13, 0))
                .active(true)
                .build();

        doctorScheduleService.replaceWeekly(doctor.getId(), List.of(updatedMonday));

        // La entidad existente se mutó y se guardó
        assertThat(mondaySchedule.getStartTime()).isEqualTo(LocalTime.of(9, 0));
        verify(doctorScheduleRepository).save(mondaySchedule);
    }

    @Test
    @DisplayName("replaceWeekly: lanza IllegalArgumentException si startTime >= endTime")
    void replaceWeekly_horasInvalidas_lanzaExcepcion() {
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        when(doctorScheduleRepository.findByDoctor(doctor)).thenReturn(List.of());

        // Hora de inicio igual a la de fin — no tiene sentido
        DoctorScheduleRequest badRequest = DoctorScheduleRequest.builder()
                .weekDay(Weekday.MONDAY)
                .startTime(LocalTime.of(10, 0))
                .endTime(LocalTime.of(10, 0))
                .build();

        assertThatThrownBy(() -> doctorScheduleService.replaceWeekly(
                doctor.getId(), List.of(badRequest)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("startTime must be before endTime");
    }

    // =========================================================================
    // getAvailability()
    // =========================================================================

    @Test
    @DisplayName("getAvailability: devuelve 4 slots libres para un lunes sin permisos ni citas")
    void getAvailability_diaLaboralSinCitas_devuelveSlotsLibres() {
        // 08:00-12:00 → 4 slots horarios (8h, 9h, 10h, 11h)
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        when(doctorScheduleRepository.findByDoctorAndWeekDay(doctor, Weekday.MONDAY))
                .thenReturn(Optional.of(mondaySchedule));
        when(doctorLeaveRepository.findByDoctorAndStatus(doctor, LeaveStatus.APPROVED))
                .thenReturn(List.of());
        when(appointmentRepository.findByDoctorAndDateTimeBetween(
                any(), any(LocalDateTime.class), any(LocalDateTime.class)))
                .thenReturn(List.of());

        DoctorAvailabilityDTO dto = doctorScheduleService.getAvailability(doctor.getId(), MONDAY);

        assertThat(dto.isWorking()).isTrue();
        assertThat(dto.isOnLeave()).isFalse();
        assertThat(dto.getSlots()).hasSize(4);
        // Todos los slots deben estar libres
        assertThat(dto.getSlots()).allMatch(s -> !s.isBooked());
    }

    @Test
    @DisplayName("getAvailability: marca todos los slots como ocupados cuando el médico está de permiso")
    void getAvailability_medicoDePermiso_todosLosSlotsBloqueados() {
        DoctorLeave leave = DoctorLeave.builder()
                .id(UUID.randomUUID())
                .doctor(doctor)
                .startDate(MONDAY)
                .endDate(MONDAY)
                .type(LeaveType.VACATION)
                .status(LeaveStatus.APPROVED)
                .reason("Vacaciones")
                .build();

        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        when(doctorScheduleRepository.findByDoctorAndWeekDay(doctor, Weekday.MONDAY))
                .thenReturn(Optional.of(mondaySchedule));
        when(doctorLeaveRepository.findByDoctorAndStatus(doctor, LeaveStatus.APPROVED))
                .thenReturn(List.of(leave));
        when(appointmentRepository.findByDoctorAndDateTimeBetween(
                any(), any(LocalDateTime.class), any(LocalDateTime.class)))
                .thenReturn(List.of());

        DoctorAvailabilityDTO dto = doctorScheduleService.getAvailability(doctor.getId(), MONDAY);

        assertThat(dto.isOnLeave()).isTrue();
        // onLeave=true hace que todos los slots queden booked aunque no haya citas
        assertThat(dto.getSlots()).allMatch(DoctorAvailabilityDTO.Slot::isBooked);
    }

    @Test
    @DisplayName("getAvailability: devuelve working=false cuando el médico no trabaja ese día")
    void getAvailability_sinHorarioEseDia_noTrabaja() {
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        // No hay horario para ese día de la semana
        when(doctorScheduleRepository.findByDoctorAndWeekDay(doctor, Weekday.MONDAY))
                .thenReturn(Optional.empty());
        when(doctorLeaveRepository.findByDoctorAndStatus(doctor, LeaveStatus.APPROVED))
                .thenReturn(List.of());

        DoctorAvailabilityDTO dto = doctorScheduleService.getAvailability(doctor.getId(), MONDAY);

        assertThat(dto.isWorking()).isFalse();
        assertThat(dto.getSlots()).isEmpty();
    }

    @Test
    @DisplayName("getAvailability: el slot de las 10h queda ocupado si hay una cita confirmada a esa hora")
    void getAvailability_conCitaConfirmada_slotCorrespondienteBloqueado() {
        Appointment confirmedAppt = Appointment.builder()
                .id(UUID.randomUUID())
                .doctor(doctor)
                .dateTime(LocalDateTime.of(2027, 3, 8, 10, 0))
                .status(AppointmentStatus.CONFIRMED)
                .build();

        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        when(doctorScheduleRepository.findByDoctorAndWeekDay(doctor, Weekday.MONDAY))
                .thenReturn(Optional.of(mondaySchedule));
        when(doctorLeaveRepository.findByDoctorAndStatus(doctor, LeaveStatus.APPROVED))
                .thenReturn(List.of());
        when(appointmentRepository.findByDoctorAndDateTimeBetween(
                any(), any(LocalDateTime.class), any(LocalDateTime.class)))
                .thenReturn(List.of(confirmedAppt));

        DoctorAvailabilityDTO dto = doctorScheduleService.getAvailability(doctor.getId(), MONDAY);

        // Solo el slot de las 10:00 debe estar ocupado
        long bookedCount = dto.getSlots().stream().filter(DoctorAvailabilityDTO.Slot::isBooked).count();
        assertThat(bookedCount).isEqualTo(1);
        assertThat(dto.getSlots().stream()
                .filter(s -> s.getTime().equals("10:00"))
                .findFirst().orElseThrow().isBooked()).isTrue();
    }
}
