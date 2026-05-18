package com.medicita.app.service.impl;

import com.medicita.app.dto.doctor.DoctorDTO;
import com.medicita.app.dto.doctor.DoctorRequest;
import com.medicita.app.entity.Doctor;
import com.medicita.app.entity.Specialty;
import com.medicita.app.entity.User;
import com.medicita.app.enums.Role;
import com.medicita.app.exception.ResourceNotFoundException;
import com.medicita.app.repository.DoctorRepository;
import com.medicita.app.repository.SpecialtyRepository;
import com.medicita.app.repository.UserRepository;
import com.medicita.app.service.UserService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/*
 * Pruebas unitarias para DoctorServiceImpl.
 * Este servicio maneja el CRUD de médicos, incluyendo la creación de su
 * usuario asociado y la validación de email y licencia médica únicos.
 * También cubre activar/desactivar (que afecta tanto al Doctor como al User).
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("DoctorServiceImpl — Pruebas unitarias")
class DoctorServiceImplTest {

    @Mock private DoctorRepository doctorRepository;
    @Mock private UserRepository userRepository;
    @Mock private SpecialtyRepository specialtyRepository;
    @Mock private PasswordEncoder passwordEncoder;
    @Mock private UserService userService;

    @InjectMocks
    private DoctorServiceImpl doctorService;

    private User doctorUser;
    private Doctor doctor;
    private Specialty specialty;
    private DoctorRequest validRequest;

    @BeforeEach
    void setUp() {
        specialty = Specialty.builder()
                .id(UUID.randomUUID())
                .name("Cardiología")
                .active(true)
                .build();

        doctorUser = User.builder()
                .id(UUID.randomUUID())
                .firstName("Luis")
                .lastName("Martínez")
                .email("luis@medicita.com")
                .role(Role.DOCTOR)
                .active(true)
                .build();

        doctor = Doctor.builder()
                .id(UUID.randomUUID())
                .user(doctorUser)
                .medicalLicense("MED-007")
                .specialty(specialty)
                .active(true)
                .build();

        validRequest = DoctorRequest.builder()
                .firstName("Luis")
                .lastName("Martínez")
                .email("luis@medicita.com")
                .password("Clave2026*")
                .medicalLicense("MED-007")
                .specialtyId(specialty.getId())
                .build();
    }

    // =========================================================================
    // findAll()
    // =========================================================================

    @Test
    @DisplayName("findAll: devuelve lista de todos los médicos")
    void findAll_devuelveTodosLosDoctores() {
        when(doctorRepository.findAll()).thenReturn(List.of(doctor));

        List<DoctorDTO> result = doctorService.findAll();

        assertThat(result).hasSize(1);
        assertThat(result.get(0).getMedicalLicense()).isEqualTo("MED-007");
    }

    // =========================================================================
    // findBySpecialty()
    // =========================================================================

    @Test
    @DisplayName("findBySpecialty: devuelve médicos de la especialidad indicada")
    void findBySpecialty_especialidadExiste_devuelveDoctores() {
        when(specialtyRepository.findById(specialty.getId())).thenReturn(Optional.of(specialty));
        when(doctorRepository.findBySpecialty(specialty)).thenReturn(List.of(doctor));

        List<DoctorDTO> result = doctorService.findBySpecialty(specialty.getId());

        assertThat(result).hasSize(1);
        assertThat(result.get(0).getSpecialtyName()).isEqualTo("Cardiología");
    }

    @Test
    @DisplayName("findBySpecialty: lanza ResourceNotFoundException si la especialidad no existe")
    void findBySpecialty_especialidadNoExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(specialtyRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> doctorService.findBySpecialty(fakeId))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    // =========================================================================
    // findById()
    // =========================================================================

    @Test
    @DisplayName("findById: devuelve el médico cuando existe")
    void findById_doctorExiste_devuelveDTO() {
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));

        DoctorDTO result = doctorService.findById(doctor.getId());

        assertThat(result.getEmail()).isEqualTo("luis@medicita.com");
    }

    @Test
    @DisplayName("findById: lanza ResourceNotFoundException si el médico no existe")
    void findById_doctorNoExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(doctorRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> doctorService.findById(fakeId))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    // =========================================================================
    // create()
    // =========================================================================

    @Test
    @DisplayName("create: crea médico con su usuario y lo persiste correctamente")
    void create_conDatosValidos_creaUsuarioYDoctor() {
        when(userRepository.existsByEmail(validRequest.getEmail())).thenReturn(false);
        when(doctorRepository.existsByMedicalLicense(validRequest.getMedicalLicense())).thenReturn(false);
        when(specialtyRepository.findById(specialty.getId())).thenReturn(Optional.of(specialty));
        when(passwordEncoder.encode(anyString())).thenReturn("$2a$hashed");
        when(doctorRepository.save(any(Doctor.class))).thenReturn(doctor);

        DoctorDTO result = doctorService.create(validRequest);

        assertThat(result).isNotNull();
        // El usuario debe haberse guardado antes que el doctor
        verify(userRepository).save(any(User.class));
        verify(doctorRepository).save(any(Doctor.class));
    }

    @Test
    @DisplayName("create: lanza RuntimeException si el email ya está registrado")
    void create_emailDuplicado_lanzaExcepcion() {
        when(userRepository.existsByEmail(validRequest.getEmail())).thenReturn(true);

        assertThatThrownBy(() -> doctorService.create(validRequest))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("Email already registered");
    }

    @Test
    @DisplayName("create: lanza RuntimeException si la licencia médica ya está registrada")
    void create_licenciaDuplicada_lanzaExcepcion() {
        when(userRepository.existsByEmail(anyString())).thenReturn(false);
        when(doctorRepository.existsByMedicalLicense(validRequest.getMedicalLicense())).thenReturn(true);

        assertThatThrownBy(() -> doctorService.create(validRequest))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("Medical license already registered");
    }

    // =========================================================================
    // update()
    // =========================================================================

    @Test
    @DisplayName("update: actualiza nombre y especialidad del médico")
    void update_conDatosValidos_actualizaDoctor() {
        DoctorRequest updateReq = DoctorRequest.builder()
                .firstName("Luis Alberto")
                .lastName("Martínez")
                .email("luis@medicita.com")
                .password("Clave2026*")
                .medicalLicense("MED-007-A")
                .specialtyId(specialty.getId())
                .build();

        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));
        when(specialtyRepository.findById(specialty.getId())).thenReturn(Optional.of(specialty));
        when(doctorRepository.save(any())).thenReturn(doctor);

        doctorService.update(doctor.getId(), updateReq);

        assertThat(doctorUser.getFirstName()).isEqualTo("Luis Alberto");
        assertThat(doctor.getMedicalLicense()).isEqualTo("MED-007-A");
        verify(userRepository).save(doctorUser);
        verify(doctorRepository).save(doctor);
    }

    // =========================================================================
    // deactivate() y activate()
    // =========================================================================

    @Test
    @DisplayName("deactivate: marca como inactivo al médico y su usuario")
    void deactivate_doctorExiste_desactivaAmbos() {
        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));

        doctorService.deactivate(doctor.getId());

        // Tanto el Doctor como su User deben quedar desactivados
        assertThat(doctor.isActive()).isFalse();
        assertThat(doctorUser.isActive()).isFalse();
        verify(doctorRepository).save(doctor);
        verify(userRepository).save(doctorUser);
    }

    @Test
    @DisplayName("activate: reactiva al médico y su usuario")
    void activate_doctorInactivo_activaAmbos() {
        doctor.setActive(false);
        doctorUser.setActive(false);

        when(doctorRepository.findById(doctor.getId())).thenReturn(Optional.of(doctor));

        doctorService.activate(doctor.getId());

        assertThat(doctor.isActive()).isTrue();
        assertThat(doctorUser.isActive()).isTrue();
    }

    @Test
    @DisplayName("deactivate: lanza ResourceNotFoundException si el médico no existe")
    void deactivate_doctorNoExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(doctorRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> doctorService.deactivate(fakeId))
                .isInstanceOf(ResourceNotFoundException.class);
    }
}
