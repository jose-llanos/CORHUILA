package com.medicita.app.service.impl;

import com.medicita.app.dto.patient.PatientDTO;
import com.medicita.app.entity.Patient;
import com.medicita.app.entity.User;
import com.medicita.app.enums.Role;
import com.medicita.app.exception.ResourceNotFoundException;
import com.medicita.app.repository.PatientRepository;
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
 * Pruebas unitarias para PatientServiceImpl.
 * Es un servicio relativamente simple: busca y actualiza pacientes.
 * No tiene mucha lógica de negocio propia, pero igual vale la pena
 * cubrir los caminos de error (cuando no encuentra al paciente).
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("PatientServiceImpl — Pruebas unitarias")
class PatientServiceImplTest {

    @Mock private PatientRepository patientRepository;
    @Mock private UserService userService;

    @InjectMocks
    private PatientServiceImpl patientService;

    private User user;
    private Patient patient;

    @BeforeEach
    void setUp() {
        user = User.builder()
                .id(UUID.randomUUID())
                .firstName("Carlos")
                .lastName("Ruiz")
                .email("carlos@medicita.com")
                .role(Role.PATIENT)
                .active(true)
                .build();

        patient = Patient.builder()
                .id(UUID.randomUUID())
                .user(user)
                .documentNumber("1075999001")
                .phone("3109876543")
                .birthDate(LocalDate.of(1990, 7, 20))
                .build();
    }

    // =========================================================================
    // findById()
    // =========================================================================

    @Test
    @DisplayName("findById: devuelve PatientDTO cuando el paciente existe")
    void findById_pacienteExiste_devuelveDTO() {
        when(patientRepository.findById(patient.getId())).thenReturn(Optional.of(patient));

        PatientDTO result = patientService.findById(patient.getId());

        assertThat(result).isNotNull();
        assertThat(result.getFirstName()).isEqualTo("Carlos");
        assertThat(result.getDocumentNumber()).isEqualTo("1075999001");
    }

    @Test
    @DisplayName("findById: lanza ResourceNotFoundException si el paciente no existe")
    void findById_pacienteNoExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(patientRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> patientService.findById(fakeId))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    // =========================================================================
    // findByCurrentUser()
    // =========================================================================

    @Test
    @DisplayName("findByCurrentUser: devuelve el perfil del usuario logueado")
    void findByCurrentUser_conUsuarioValido_devuelveDTO() {
        when(userService.getCurrentUser()).thenReturn(user);
        when(patientRepository.findByUser(user)).thenReturn(Optional.of(patient));

        PatientDTO result = patientService.findByCurrentUser();

        assertThat(result.getEmail()).isEqualTo("carlos@medicita.com");
    }

    @Test
    @DisplayName("findByCurrentUser: lanza ResourceNotFoundException si el usuario no tiene perfil")
    void findByCurrentUser_sinPerfil_lanzaExcepcion() {
        when(userService.getCurrentUser()).thenReturn(user);
        when(patientRepository.findByUser(user)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> patientService.findByCurrentUser())
                .isInstanceOf(ResourceNotFoundException.class)
                .hasMessageContaining("Patient profile not found");
    }

    // =========================================================================
    // findAll()
    // =========================================================================

    @Test
    @DisplayName("findAll: devuelve lista completa de pacientes")
    void findAll_devuelveTodosLosPacientes() {
        when(patientRepository.findAll()).thenReturn(List.of(patient));

        List<PatientDTO> result = patientService.findAll();

        assertThat(result).hasSize(1);
        assertThat(result.get(0).getDocumentNumber()).isEqualTo("1075999001");
    }

    @Test
    @DisplayName("findAll: devuelve lista vacía cuando no hay pacientes")
    void findAll_sinPacientes_devuelveListaVacia() {
        when(patientRepository.findAll()).thenReturn(List.of());

        assertThat(patientService.findAll()).isEmpty();
    }

    // =========================================================================
    // update()
    // =========================================================================

    @Test
    @DisplayName("update: actualiza los datos del paciente correctamente")
    void update_conDatosValidos_actualizaYDevuelveDTO() {
        PatientDTO updateData = PatientDTO.builder()
                .phone("3001112233")
                .birthDate(LocalDate.of(1990, 7, 20))
                .documentNumber("1075000099")
                .build();

        when(patientRepository.findById(patient.getId())).thenReturn(Optional.of(patient));
        when(patientRepository.save(any(Patient.class))).thenReturn(patient);

        PatientDTO result = patientService.update(patient.getId(), updateData);

        // Verificamos que se guardó y que los datos cambiaron en la entidad
        verify(patientRepository).save(patient);
        assertThat(patient.getPhone()).isEqualTo("3001112233");
        assertThat(patient.getDocumentNumber()).isEqualTo("1075000099");
        assertThat(result).isNotNull();
    }

    @Test
    @DisplayName("update: lanza ResourceNotFoundException si el paciente no existe")
    void update_pacienteNoExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(patientRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> patientService.update(fakeId, new PatientDTO()))
                .isInstanceOf(ResourceNotFoundException.class);
    }
}
