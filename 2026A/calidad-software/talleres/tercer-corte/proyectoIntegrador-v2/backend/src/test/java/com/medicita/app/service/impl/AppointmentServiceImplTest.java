package com.medicita.app.service.impl;

import com.medicita.app.dto.appointment.AppointmentDTO;
import com.medicita.app.dto.appointment.AppointmentRequest;
import com.medicita.app.dto.appointment.AppointmentStatusUpdateRequest;
import com.medicita.app.entity.*;
import com.medicita.app.enums.AppointmentStatus;
import com.medicita.app.enums.LeaveStatus;
import com.medicita.app.enums.LeaveType;
import com.medicita.app.enums.Role;
import com.medicita.app.enums.Weekday;
import com.medicita.app.exception.ResourceNotFoundException;
import com.medicita.app.repository.*;
import com.medicita.app.service.UserService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.PageRequest;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

/*
 * Pruebas unitarias para AppointmentServiceImpl.
 *
 * Este es el servicio más crítico del sistema: maneja toda la lógica de
 * agendamiento de citas. Tiene varias validaciones encadenadas antes de
 * crear una cita (doctor activo, horario disponible, sin conflictos, sin
 * permisos aprobados), así que los casos de prueba son bastante variados.
 *
 * Igual que en AuthServiceImpl, usamos Mockito para no depender de la BD real.
 * La fecha de prueba usada es 2027-03-08 (lunes) para que el Weekday.MONDAY
 * coincida sin magia negra.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("AppointmentServiceImpl — Pruebas unitarias")
class AppointmentServiceImplTest {

    @Mock private AppointmentRepository appointmentRepository;
    @Mock private PatientRepository patientRepository;
    @Mock private DoctorRepository doctorRepository;
    @Mock private DoctorLeaveRepository doctorLeaveRepository;
    @Mock private DoctorScheduleRepository doctorScheduleRepository;
    @Mock private UserService userService;

    @InjectMocks
    private AppointmentServiceImpl appointmentService;

    // Entidades de prueba reutilizadas en múltiples tests
    private User patientUser;
    private User doctorUser;
    private Patient patient;
    private Doctor doctor;
    private DoctorSchedule mondaySchedule;
    private AppointmentRequest validRequest;

    // Lunes 8 de marzo de 2027 a las 10:00 — dentro del horario 08:00-17:00
    private static final LocalDateTime VALID_DATE_TIME = LocalDateTime.of(2027, 3, 8, 10, 0);

    @BeforeEach
    void setUp() {
        // Usuario del paciente
        patientUser = User.builder()
                .id(UUID.randomUUID())
                .firstName("Juan")
                .lastName("Pérez")
                .email("juan@medicita.com")
                .role(Role.PATIENT)
                .active(true)
                .build();

        // Usuario del doctor
        doctorUser = User.builder()
                .id(UUID.randomUUID())
                .firstName("Ana")
                .lastName("García")
                .email("ana@medicita.com")
                .role(Role.DOCTOR)
                .active(true)
                .build();

        patient = Patient.builder()
                .id(UUID.randomUUID())
                .user(patientUser)
                .documentNumber("1075000001")
                .build();

        doctor = Doctor.builder()
                .id(UUID.randomUUID())
                .user(doctorUser)
                .medicalLicense("MED-001")
                .active(true)
                .build();

        // El doctor trabaja los lunes de 08:00 a 17:00
        mondaySchedule = DoctorSchedule.builder()
                .id(UUID.randomUUID())
                .doctor(doctor)
                .weekDay(Weekday.MONDAY)
                .startTime(LocalTime.of(8, 0))
                .endTime(LocalTime.of(17, 0))
                .active(true)
                .build();

        validRequest = AppointmentRequest.builder()
                .doctorId(doctor.getId())
                .appointmentDateTime(VALID_DATE_TIME)
                .reason("Control general")
                .build();
    }

    // =========================================================================
    // Pruebas de create()
    // =========================================================================

    /*
     * Caso feliz: todas las validaciones pasan y se guarda la cita.
     * Doctor activo, horario disponible, sin conflicto de hora, sin permiso aprobado.
     */
    @Test
    @DisplayName("create: crea cita exitosamente cuando todas las validaciones pasan")
    void create_conDatosValidos_guardaCitaYDevuelveDTO() {
        when(userService.getCurrentUser()).thenReturn(patientUser);
        when(patientRepository.findByUser(patientUser)).thenReturn(Optional.of(patient));
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        when(appointmentRepository.existsByDoctorAndDateTime(eq(doctor), eq(VALID_DATE_TIME))).thenReturn(false);
        when(doctorScheduleRepository.findByDoctorAndWeekDay(eq(doctor), eq(Weekday.MONDAY)))
                .thenReturn(Optional.of(mondaySchedule));
        when(doctorLeaveRepository.findByDoctorAndStatus(eq(doctor), eq(LeaveStatus.APPROVED)))
                .thenReturn(List.of());

        // El save devuelve una cita con id para simular lo que haría la BD
        Appointment saved = Appointment.builder()
                .id(UUID.randomUUID())
                .patient(patient)
                .doctor(doctor)
                .dateTime(VALID_DATE_TIME)
                .reason("Control general")
                .status(AppointmentStatus.PENDING)
                .build();
        when(appointmentRepository.save(any(Appointment.class))).thenReturn(saved);

        AppointmentDTO result = appointmentService.create(validRequest);

        assertThat(result).isNotNull();
        assertThat(result.getId()).isEqualTo(saved.getId());
        assertThat(result.getDoctorFullName()).isEqualTo("Ana García");
        assertThat(result.getPatientFullName()).isEqualTo("Juan Pérez");
        assertThat(result.getStatus()).isEqualTo("PENDING");
        verify(appointmentRepository).save(any(Appointment.class));
    }

    /*
     * Si el usuario logueado no tiene perfil de paciente, se lanza excepción.
     * Esto puede pasar si un admin intenta agendar sin tener paciente asociado.
     */
    @Test
    @DisplayName("create: lanza ResourceNotFoundException si no existe perfil de paciente")
    void create_sinPerfilPaciente_lanzaResourceNotFoundException() {
        when(userService.getCurrentUser()).thenReturn(patientUser);
        when(patientRepository.findByUser(patientUser)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> appointmentService.create(validRequest))
                .isInstanceOf(ResourceNotFoundException.class)
                .hasMessageContaining("Patient profile not found");

        // Si no encontró al paciente, no tiene sentido buscar el doctor
        verify(doctorRepository, never()).findById(any());
    }

    /*
     * Error: el doctor solicitado no existe en el sistema.
     */
    @Test
    @DisplayName("create: lanza ResourceNotFoundException si el doctor no existe")
    void create_doctorNoExiste_lanzaResourceNotFoundException() {
        when(userService.getCurrentUser()).thenReturn(patientUser);
        when(patientRepository.findByUser(patientUser)).thenReturn(Optional.of(patient));
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.empty());

        assertThatThrownBy(() -> appointmentService.create(validRequest))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    /*
     * Error: el doctor existe pero está desactivado. No se le pueden agendar citas.
     */
    @Test
    @DisplayName("create: lanza RuntimeException si el doctor está inactivo")
    void create_doctorInactivo_lanzaRuntimeException() {
        doctor.setActive(false);

        when(userService.getCurrentUser()).thenReturn(patientUser);
        when(patientRepository.findByUser(patientUser)).thenReturn(Optional.of(patient));
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));

        assertThatThrownBy(() -> appointmentService.create(validRequest))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("Doctor is not currently active");
    }

    /*
     * Error: el doctor ya tiene otra cita exactamente en ese mismo horario.
     * Evita doble-reserva para el mismo slot.
     */
    @Test
    @DisplayName("create: lanza RuntimeException si el doctor ya tiene cita en ese horario")
    void create_horarioOcupado_lanzaRuntimeException() {
        when(userService.getCurrentUser()).thenReturn(patientUser);
        when(patientRepository.findByUser(patientUser)).thenReturn(Optional.of(patient));
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        when(appointmentRepository.existsByDoctorAndDateTime(eq(doctor), eq(VALID_DATE_TIME))).thenReturn(true);

        assertThatThrownBy(() -> appointmentService.create(validRequest))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("ya tiene una cita en ese horario");
    }

    /*
     * Error: el doctor no tiene horario configurado para ese día de la semana.
     * No tiene sentido agendar el lunes si el doctor no trabaja los lunes.
     */
    @Test
    @DisplayName("create: lanza RuntimeException si el doctor no trabaja ese día")
    void create_sinHorarioEseDia_lanzaRuntimeException() {
        when(userService.getCurrentUser()).thenReturn(patientUser);
        when(patientRepository.findByUser(patientUser)).thenReturn(Optional.of(patient));
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        when(appointmentRepository.existsByDoctorAndDateTime(any(), any())).thenReturn(false);
        // No hay horario para ese día
        when(doctorScheduleRepository.findByDoctorAndWeekDay(eq(doctor), eq(Weekday.MONDAY)))
                .thenReturn(Optional.empty());

        assertThatThrownBy(() -> appointmentService.create(validRequest))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("no atiende ese día");
    }

    /*
     * Caso límite: el horario del día existe pero está desactivado.
     * Mismo resultado que no tener horario — el médico no atiende.
     */
    @Test
    @DisplayName("create: lanza RuntimeException si el horario del día está desactivado")
    void create_horarioDiaDesactivado_lanzaRuntimeException() {
        mondaySchedule.setActive(false);

        when(userService.getCurrentUser()).thenReturn(patientUser);
        when(patientRepository.findByUser(patientUser)).thenReturn(Optional.of(patient));
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        when(appointmentRepository.existsByDoctorAndDateTime(any(), any())).thenReturn(false);
        when(doctorScheduleRepository.findByDoctorAndWeekDay(any(), any()))
                .thenReturn(Optional.of(mondaySchedule));

        assertThatThrownBy(() -> appointmentService.create(validRequest))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("no atiende ese día");
    }

    /*
     * Caso límite: la hora de la cita cae fuera del horario laboral del médico.
     * Por ejemplo, agendar a las 07:00 cuando el médico empieza a las 08:00.
     */
    @Test
    @DisplayName("create: lanza RuntimeException si la hora está fuera del horario laboral")
    void create_horaFueraDeHorario_lanzaRuntimeException() {
        // Intentamos agendar a las 07:00, antes de que empiece el turno (08:00)
        AppointmentRequest earlyRequest = AppointmentRequest.builder()
                .doctorId(doctor.getId())
                .appointmentDateTime(LocalDateTime.of(2027, 3, 8, 7, 0))
                .reason("Urgencia")
                .build();

        when(userService.getCurrentUser()).thenReturn(patientUser);
        when(patientRepository.findByUser(patientUser)).thenReturn(Optional.of(patient));
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        when(appointmentRepository.existsByDoctorAndDateTime(any(), any())).thenReturn(false);
        when(doctorScheduleRepository.findByDoctorAndWeekDay(any(), any()))
                .thenReturn(Optional.of(mondaySchedule));

        assertThatThrownBy(() -> appointmentService.create(earlyRequest))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("fuera del horario laboral");
    }

    /*
     * Error: el médico tiene un permiso aprobado que cubre la fecha de la cita.
     * No se puede agendar si el médico está de vacaciones o incapacitado.
     */
    @Test
    @DisplayName("create: lanza RuntimeException si el médico está en permiso aprobado")
    void create_medicoEnPermiso_lanzaRuntimeException() {
        // Permiso aprobado que cubre la semana de la cita
        DoctorLeave leave = DoctorLeave.builder()
                .id(UUID.randomUUID())
                .doctor(doctor)
                .startDate(LocalDate.of(2027, 3, 7))
                .endDate(LocalDate.of(2027, 3, 14))
                .status(LeaveStatus.APPROVED)
                .type(LeaveType.VACATION)
                .build();

        when(userService.getCurrentUser()).thenReturn(patientUser);
        when(patientRepository.findByUser(patientUser)).thenReturn(Optional.of(patient));
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        when(appointmentRepository.existsByDoctorAndDateTime(any(), any())).thenReturn(false);
        when(doctorScheduleRepository.findByDoctorAndWeekDay(any(), any()))
                .thenReturn(Optional.of(mondaySchedule));
        when(doctorLeaveRepository.findByDoctorAndStatus(eq(doctor), eq(LeaveStatus.APPROVED)))
                .thenReturn(List.of(leave));

        assertThatThrownBy(() -> appointmentService.create(validRequest))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("permiso aprobado");
    }

    // =========================================================================
    // Pruebas de updateStatus()
    // =========================================================================

    /*
     * Caso feliz: se actualiza el estado de la cita. Además se verifica que
     * las notas también se guarden cuando vienen en el request.
     */
    @Test
    @DisplayName("updateStatus: actualiza estado y notas de la cita correctamente")
    void updateStatus_conDatosValidos_actualizaYDevuelveDTO() {
        UUID appointmentId = UUID.randomUUID();
        Appointment appointment = Appointment.builder()
                .id(appointmentId)
                .patient(patient)
                .doctor(doctor)
                .dateTime(VALID_DATE_TIME)
                .status(AppointmentStatus.PENDING)
                .build();

        AppointmentStatusUpdateRequest updateRequest = new AppointmentStatusUpdateRequest(
                AppointmentStatus.COMPLETED, "Paciente atendido sin novedades"
        );

        when(appointmentRepository.findById(appointmentId)).thenReturn(Optional.of(appointment));
        when(appointmentRepository.save(any(Appointment.class))).thenReturn(appointment);

        AppointmentDTO result = appointmentService.updateStatus(appointmentId, updateRequest);

        // Verificamos que el estado y las notas quedaron aplicados
        assertThat(appointment.getStatus()).isEqualTo(AppointmentStatus.COMPLETED);
        assertThat(appointment.getNotes()).isEqualTo("Paciente atendido sin novedades");
        assertThat(result).isNotNull();
        verify(appointmentRepository).save(appointment);
    }

    /*
     * Si se actualiza el estado sin notas, el campo notes no debe pisarse.
     */
    @Test
    @DisplayName("updateStatus: no modifica notas si el campo viene null")
    void updateStatus_sinNotas_noModificaNotasExistentes() {
        UUID appointmentId = UUID.randomUUID();
        Appointment appointment = Appointment.builder()
                .id(appointmentId)
                .patient(patient)
                .doctor(doctor)
                .dateTime(VALID_DATE_TIME)
                .status(AppointmentStatus.PENDING)
                .notes("Notas previas")
                .build();

        // El request trae null en notes
        AppointmentStatusUpdateRequest updateRequest = new AppointmentStatusUpdateRequest(
                AppointmentStatus.CANCELLED, null
        );

        when(appointmentRepository.findById(appointmentId)).thenReturn(Optional.of(appointment));
        when(appointmentRepository.save(any())).thenReturn(appointment);

        appointmentService.updateStatus(appointmentId, updateRequest);

        // Las notas anteriores deben seguir intactas
        assertThat(appointment.getNotes()).isEqualTo("Notas previas");
    }

    /*
     * Error: se intenta actualizar una cita que no existe en la BD.
     */
    @Test
    @DisplayName("updateStatus: lanza ResourceNotFoundException si la cita no existe")
    void updateStatus_citaNoExiste_lanzaResourceNotFoundException() {
        UUID fakeId = UUID.randomUUID();
        when(appointmentRepository.findById(fakeId)).thenReturn(Optional.empty());

        AppointmentStatusUpdateRequest req = new AppointmentStatusUpdateRequest(
                AppointmentStatus.CANCELLED, null
        );

        assertThatThrownBy(() -> appointmentService.updateStatus(fakeId, req))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    // =========================================================================
    // Pruebas de cancel()
    // =========================================================================

    /*
     * Caso feliz: el paciente que creó la cita la cancela. El estado debe
     * quedar en CANCELLED.
     */
    @Test
    @DisplayName("cancel: el paciente cancela su propia cita exitosamente")
    void cancel_pacienteCancelaSuPropiaCita_cambiaEstadoACancelled() {
        UUID appointmentId = UUID.randomUUID();
        Appointment appointment = Appointment.builder()
                .id(appointmentId)
                .patient(patient)
                .doctor(doctor)
                .dateTime(VALID_DATE_TIME)
                .status(AppointmentStatus.PENDING)
                .build();

        // El usuario actual es el mismo que tiene la cita
        when(appointmentRepository.findById(appointmentId)).thenReturn(Optional.of(appointment));
        when(userService.getCurrentUser()).thenReturn(patientUser);

        appointmentService.cancel(appointmentId);

        ArgumentCaptor<Appointment> captor = ArgumentCaptor.forClass(Appointment.class);
        verify(appointmentRepository).save(captor.capture());
        assertThat(captor.getValue().getStatus()).isEqualTo(AppointmentStatus.CANCELLED);
    }

    /*
     * Error: un paciente diferente intenta cancelar la cita de otro.
     * Esto es una violación de seguridad — cada quien solo puede cancelar las suyas.
     */
    @Test
    @DisplayName("cancel: lanza RuntimeException si otro paciente intenta cancelar la cita")
    void cancel_otroPacienteIntentaCancelar_lanzaRuntimeException() {
        UUID appointmentId = UUID.randomUUID();
        Appointment appointment = Appointment.builder()
                .id(appointmentId)
                .patient(patient) // la cita es del patientUser
                .doctor(doctor)
                .dateTime(VALID_DATE_TIME)
                .status(AppointmentStatus.PENDING)
                .build();

        // Un usuario diferente (otro UUID) está logueado
        User otherUser = User.builder()
                .id(UUID.randomUUID()) // UUID distinto al del paciente
                .firstName("Pedro")
                .lastName("López")
                .role(Role.PATIENT)
                .active(true)
                .build();

        when(appointmentRepository.findById(appointmentId)).thenReturn(Optional.of(appointment));
        when(userService.getCurrentUser()).thenReturn(otherUser);

        assertThatThrownBy(() -> appointmentService.cancel(appointmentId))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("Only the patient who made this appointment");

        // La cita no debe haberse guardado con estado cancelado
        verify(appointmentRepository, never()).save(any());
    }

    /*
     * Error: se intenta cancelar una cita que no existe.
     */
    @Test
    @DisplayName("cancel: lanza ResourceNotFoundException si la cita no existe")
    void cancel_citaNoExiste_lanzaResourceNotFoundException() {
        UUID fakeId = UUID.randomUUID();
        when(appointmentRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> appointmentService.cancel(fakeId))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    // =========================================================================
    // Pruebas de findByCurrentPatient() y findByCurrentDoctor()
    // =========================================================================

    /*
     * Verifica que el paciente logueado recibe su propia página de citas.
     */
    @Test
    @DisplayName("findByCurrentPatient: devuelve las citas del paciente logueado")
    void findByCurrentPatient_conPacienteValido_devuelvePaginaDeCitas() {
        Appointment appointment = Appointment.builder()
                .id(UUID.randomUUID())
                .patient(patient)
                .doctor(doctor)
                .dateTime(VALID_DATE_TIME)
                .status(AppointmentStatus.PENDING)
                .build();

        PageRequest pageable = PageRequest.of(0, 10);
        Page<Appointment> page = new PageImpl<>(List.of(appointment));

        when(userService.getCurrentUser()).thenReturn(patientUser);
        when(patientRepository.findByUser(patientUser)).thenReturn(Optional.of(patient));
        when(appointmentRepository.findByPatient(eq(patient), eq(pageable))).thenReturn(page);

        Page<AppointmentDTO> result = appointmentService.findByCurrentPatient(pageable);

        assertThat(result).isNotNull();
        assertThat(result.getContent()).hasSize(1);
    }

    /*
     * Error: el usuario logueado como paciente no tiene perfil de paciente creado.
     */
    @Test
    @DisplayName("findByCurrentPatient: lanza ResourceNotFoundException si no hay perfil")
    void findByCurrentPatient_sinPerfilPaciente_lanzaResourceNotFoundException() {
        when(userService.getCurrentUser()).thenReturn(patientUser);
        when(patientRepository.findByUser(patientUser)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> appointmentService.findByCurrentPatient(PageRequest.of(0, 10)))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    /*
     * Verifica que el doctor logueado recibe su propia lista de citas.
     */
    @Test
    @DisplayName("findByCurrentDoctor: devuelve las citas del doctor logueado")
    void findByCurrentDoctor_conDoctorValido_devuelveListaDeCitas() {
        Appointment appointment = Appointment.builder()
                .id(UUID.randomUUID())
                .patient(patient)
                .doctor(doctor)
                .dateTime(VALID_DATE_TIME)
                .status(AppointmentStatus.PENDING)
                .build();

        when(userService.getCurrentUser()).thenReturn(doctorUser);
        when(doctorRepository.findByUser(doctorUser)).thenReturn(Optional.of(doctor));
        when(appointmentRepository.findByDoctor(doctor)).thenReturn(List.of(appointment));

        List<AppointmentDTO> result = appointmentService.findByCurrentDoctor();

        assertThat(result).hasSize(1);
        assertThat(result.get(0).getDoctorFullName()).isEqualTo("Ana García");
    }

    /*
     * findAll() debe devolver todas las citas sin filtro. Lo usa el módulo admin.
     */
    @Test
    @DisplayName("findAll: devuelve todas las citas del sistema")
    void findAll_devuelveTodoSinFiltro() {
        Appointment a1 = Appointment.builder().id(UUID.randomUUID())
                .patient(patient).doctor(doctor).dateTime(VALID_DATE_TIME)
                .status(AppointmentStatus.PENDING).build();
        Appointment a2 = Appointment.builder().id(UUID.randomUUID())
                .patient(patient).doctor(doctor).dateTime(VALID_DATE_TIME.plusDays(1))
                .status(AppointmentStatus.COMPLETED).build();

        when(appointmentRepository.findAll()).thenReturn(List.of(a1, a2));

        List<AppointmentDTO> result = appointmentService.findAll();

        assertThat(result).hasSize(2);
    }
}
