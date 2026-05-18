package com.medicita.app.service.impl;

import com.medicita.app.dto.leave.DoctorLeaveDTO;
import com.medicita.app.dto.leave.DoctorLeaveRequest;
import com.medicita.app.entity.Doctor;
import com.medicita.app.entity.DoctorLeave;
import com.medicita.app.entity.User;
import com.medicita.app.enums.LeaveStatus;
import com.medicita.app.enums.LeaveType;
import com.medicita.app.enums.Role;
import com.medicita.app.exception.ResourceNotFoundException;
import com.medicita.app.repository.DoctorLeaveRepository;
import com.medicita.app.repository.DoctorRepository;
import com.medicita.app.service.UserService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/*
 * Pruebas unitarias para DoctorLeaveServiceImpl.
 * Maneja las solicitudes de permiso de los médicos (vacaciones, incapacidades, etc.)
 * y su aprobación/rechazo por parte del admin. El punto más interesante
 * es la validación de fechas solapadas: no se puede pedir dos permisos
 * que se crucen en el mismo rango de fechas.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("DoctorLeaveServiceImpl — Pruebas unitarias")
class DoctorLeaveServiceImplTest {

    @Mock private DoctorLeaveRepository doctorLeaveRepository;
    @Mock private DoctorRepository doctorRepository;
    @Mock private UserService userService;

    @InjectMocks
    private DoctorLeaveServiceImpl doctorLeaveService;

    private User doctorUser;
    private Doctor doctor;
    private DoctorLeave leave;
    private DoctorLeaveRequest validRequest;

    @BeforeEach
    void setUp() {
        doctorUser = User.builder()
                .id(UUID.randomUUID())
                .firstName("Sofía")
                .lastName("Torres")
                .email("sofia@medicita.com")
                .role(Role.DOCTOR)
                .active(true)
                .build();

        doctor = Doctor.builder()
                .id(UUID.randomUUID())
                .user(doctorUser)
                .medicalLicense("MED-020")
                .active(true)
                .build();

        leave = DoctorLeave.builder()
                .id(UUID.randomUUID())
                .doctor(doctor)
                .startDate(LocalDate.of(2027, 6, 1))
                .endDate(LocalDate.of(2027, 6, 7))
                .type(LeaveType.VACATION)
                .status(LeaveStatus.PENDING)
                .reason("Vacaciones anuales")
                .build();

        validRequest = DoctorLeaveRequest.builder()
                .startDate(LocalDate.of(2027, 6, 1))
                .endDate(LocalDate.of(2027, 6, 7))
                .type(LeaveType.VACATION)
                .reason("Vacaciones anuales")
                .build();
    }

    // =========================================================================
    // requestLeave()
    // =========================================================================

    @Test
    @DisplayName("requestLeave: crea el permiso cuando las fechas son válidas y no hay solapamiento")
    void requestLeave_fechasValidasSinSolapamiento_creaPermiso() {
        when(userService.getCurrentUser()).thenReturn(doctorUser);
        when(doctorRepository.findByUser(doctorUser)).thenReturn(Optional.of(doctor));
        when(doctorLeaveRepository.existsByDoctorAndStartDateLessThanEqualAndEndDateGreaterThanEqual(
                any(), any(), any())).thenReturn(false);
        when(doctorLeaveRepository.save(any(DoctorLeave.class))).thenReturn(leave);

        DoctorLeaveDTO result = doctorLeaveService.requestLeave(validRequest);

        assertThat(result).isNotNull();
        assertThat(result.getStatus()).isEqualTo("PENDING");
        verify(doctorLeaveRepository).save(any(DoctorLeave.class));
    }

    @Test
    @DisplayName("requestLeave: lanza RuntimeException si la fecha fin es anterior a la fecha inicio")
    void requestLeave_fechaFinAntesDeInicio_lanzaExcepcion() {
        // Fin antes que inicio — no tiene sentido calendáricamente
        DoctorLeaveRequest badRequest = DoctorLeaveRequest.builder()
                .startDate(LocalDate.of(2027, 6, 10))
                .endDate(LocalDate.of(2027, 6, 5))
                .type(LeaveType.VACATION)
                .reason("Error de fechas")
                .build();

        assertThatThrownBy(() -> doctorLeaveService.requestLeave(badRequest))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("End date must be after or equal to start date");
    }

    @Test
    @DisplayName("requestLeave: lanza RuntimeException si ya existe un permiso solapado")
    void requestLeave_conSolapamiento_lanzaExcepcion() {
        when(userService.getCurrentUser()).thenReturn(doctorUser);
        when(doctorRepository.findByUser(doctorUser)).thenReturn(Optional.of(doctor));
        // Ya hay un permiso que se cruza con esas fechas
        when(doctorLeaveRepository.existsByDoctorAndStartDateLessThanEqualAndEndDateGreaterThanEqual(
                any(), any(), any())).thenReturn(true);

        assertThatThrownBy(() -> doctorLeaveService.requestLeave(validRequest))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("leave request already exists overlapping");
    }

    @Test
    @DisplayName("requestLeave: lanza ResourceNotFoundException si el usuario no tiene perfil de doctor")
    void requestLeave_sinPerfilDoctor_lanzaExcepcion() {
        when(userService.getCurrentUser()).thenReturn(doctorUser);
        when(doctorRepository.findByUser(doctorUser)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> doctorLeaveService.requestLeave(validRequest))
                .isInstanceOf(ResourceNotFoundException.class)
                .hasMessageContaining("Doctor profile not found");
    }

    // =========================================================================
    // findByCurrentDoctor() y findApprovedByCurrentDoctor()
    // =========================================================================

    @Test
    @DisplayName("findByCurrentDoctor: devuelve todos los permisos del doctor logueado")
    void findByCurrentDoctor_conDoctorValido_devuelveListaCompleta() {
        when(userService.getCurrentUser()).thenReturn(doctorUser);
        when(doctorRepository.findByUser(doctorUser)).thenReturn(Optional.of(doctor));
        when(doctorLeaveRepository.findByDoctor(doctor)).thenReturn(List.of(leave));

        List<DoctorLeaveDTO> result = doctorLeaveService.findByCurrentDoctor();

        assertThat(result).hasSize(1);
        assertThat(result.get(0).getDoctorFullName()).isEqualTo("Sofía Torres");
    }

    @Test
    @DisplayName("findApprovedByCurrentDoctor: devuelve solo los permisos aprobados del doctor")
    void findApprovedByCurrentDoctor_devuelveSoloAprobados() {
        leave.setStatus(LeaveStatus.APPROVED);

        when(userService.getCurrentUser()).thenReturn(doctorUser);
        when(doctorRepository.findByUser(doctorUser)).thenReturn(Optional.of(doctor));
        when(doctorLeaveRepository.findByDoctorAndStatus(doctor, LeaveStatus.APPROVED))
                .thenReturn(List.of(leave));

        List<DoctorLeaveDTO> result = doctorLeaveService.findApprovedByCurrentDoctor();

        assertThat(result).hasSize(1);
        assertThat(result.get(0).getStatus()).isEqualTo("APPROVED");
    }

    // =========================================================================
    // approve() y reject()
    // =========================================================================

    @Test
    @DisplayName("approve: cambia el estado del permiso a APPROVED")
    void approve_permisoExiste_cambiaEstadoAApproved() {
        when(doctorLeaveRepository.findById(leave.getId())).thenReturn(Optional.of(leave));
        when(doctorLeaveRepository.save(any())).thenReturn(leave);

        DoctorLeaveDTO result = doctorLeaveService.approve(leave.getId());

        assertThat(leave.getStatus()).isEqualTo(LeaveStatus.APPROVED);
        assertThat(result).isNotNull();
        verify(doctorLeaveRepository).save(leave);
    }

    @Test
    @DisplayName("reject: cambia el estado del permiso a REJECTED")
    void reject_permisoExiste_cambiaEstadoARejected() {
        when(doctorLeaveRepository.findById(leave.getId())).thenReturn(Optional.of(leave));
        when(doctorLeaveRepository.save(any())).thenReturn(leave);

        DoctorLeaveDTO result = doctorLeaveService.reject(leave.getId());

        assertThat(leave.getStatus()).isEqualTo(LeaveStatus.REJECTED);
        assertThat(result).isNotNull();
    }

    @Test
    @DisplayName("approve: lanza ResourceNotFoundException si el permiso no existe")
    void approve_permisoNoExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(doctorLeaveRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> doctorLeaveService.approve(fakeId))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    // =========================================================================
    // findPending() y findAll()
    // =========================================================================

    @Test
    @DisplayName("findPending: devuelve solo los permisos en estado PENDING")
    void findPending_devuelveSoloPendientes() {
        when(doctorLeaveRepository.findByStatus(LeaveStatus.PENDING)).thenReturn(List.of(leave));

        List<DoctorLeaveDTO> result = doctorLeaveService.findPending();

        assertThat(result).hasSize(1);
        assertThat(result.get(0).getStatus()).isEqualTo("PENDING");
    }

    @Test
    @DisplayName("findAll: devuelve todos los permisos sin filtro")
    void findAll_devuelveTodosLosPermisos() {
        DoctorLeave leave2 = DoctorLeave.builder()
                .id(UUID.randomUUID())
                .doctor(doctor)
                .startDate(LocalDate.of(2027, 8, 1))
                .endDate(LocalDate.of(2027, 8, 3))
                .type(LeaveType.SICK_LEAVE)
                .status(LeaveStatus.APPROVED)
                .reason("Incapacidad médica")
                .build();

        when(doctorLeaveRepository.findAll()).thenReturn(List.of(leave, leave2));

        List<DoctorLeaveDTO> result = doctorLeaveService.findAll();

        assertThat(result).hasSize(2);
    }
}
