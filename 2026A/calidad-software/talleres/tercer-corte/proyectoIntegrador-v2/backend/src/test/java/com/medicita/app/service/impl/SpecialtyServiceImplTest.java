package com.medicita.app.service.impl;

import com.medicita.app.dto.specialty.SpecialtyDTO;
import com.medicita.app.dto.specialty.SpecialtyRequest;
import com.medicita.app.entity.Specialty;
import com.medicita.app.exception.ResourceNotFoundException;
import com.medicita.app.repository.SpecialtyRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.modelmapper.ModelMapper;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/*
 * Pruebas unitarias para SpecialtyServiceImpl.
 * Este servicio usa ModelMapper para las conversiones entre entidad y DTO,
 * así que también lo mockeamos. No es el servicio más complejo del sistema
 * pero sí tiene validación de nombre único al crear.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("SpecialtyServiceImpl — Pruebas unitarias")
class SpecialtyServiceImplTest {

    @Mock private SpecialtyRepository specialtyRepository;
    @Mock private ModelMapper modelMapper;

    @InjectMocks
    private SpecialtyServiceImpl specialtyService;

    private Specialty specialty;
    private SpecialtyDTO specialtyDTO;
    private SpecialtyRequest validRequest;

    @BeforeEach
    void setUp() {
        specialty = Specialty.builder()
                .id(UUID.randomUUID())
                .name("Pediatría")
                .description("Medicina infantil")
                .active(true)
                .build();

        specialtyDTO = SpecialtyDTO.builder()
                .id(specialty.getId())
                .name("Pediatría")
                .description("Medicina infantil")
                .build();

        validRequest = SpecialtyRequest.builder()
                .name("Pediatría")
                .description("Medicina infantil")
                .build();
    }

    // =========================================================================
    // findAll() y findAllActive()
    // =========================================================================

    @Test
    @DisplayName("findAll: devuelve todas las especialidades sin importar el estado")
    void findAll_devuelveTodas() {
        when(specialtyRepository.findAll()).thenReturn(List.of(specialty));
        when(modelMapper.map(specialty, SpecialtyDTO.class)).thenReturn(specialtyDTO);

        List<SpecialtyDTO> result = specialtyService.findAll();

        assertThat(result).hasSize(1);
        assertThat(result.get(0).getName()).isEqualTo("Pediatría");
    }

    @Test
    @DisplayName("findAllActive: devuelve solo las especialidades activas")
    void findAllActive_devuelveSoloActivas() {
        when(specialtyRepository.findByActiveTrue()).thenReturn(List.of(specialty));
        when(modelMapper.map(specialty, SpecialtyDTO.class)).thenReturn(specialtyDTO);

        List<SpecialtyDTO> result = specialtyService.findAllActive();

        assertThat(result).hasSize(1);
    }

    // =========================================================================
    // findById()
    // =========================================================================

    @Test
    @DisplayName("findById: devuelve el DTO cuando la especialidad existe")
    void findById_especialidadExiste_devuelveDTO() {
        when(specialtyRepository.findById(specialty.getId())).thenReturn(Optional.of(specialty));
        when(modelMapper.map(specialty, SpecialtyDTO.class)).thenReturn(specialtyDTO);

        SpecialtyDTO result = specialtyService.findById(specialty.getId());

        assertThat(result.getName()).isEqualTo("Pediatría");
    }

    @Test
    @DisplayName("findById: lanza ResourceNotFoundException si la especialidad no existe")
    void findById_noExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(specialtyRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> specialtyService.findById(fakeId))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    // =========================================================================
    // create()
    // =========================================================================

    @Test
    @DisplayName("create: crea la especialidad cuando el nombre es único")
    void create_nombreUnico_guardaYDevuelveDTO() {
        when(specialtyRepository.existsByName("Pediatría")).thenReturn(false);
        when(modelMapper.map(validRequest, Specialty.class)).thenReturn(specialty);
        when(specialtyRepository.save(specialty)).thenReturn(specialty);
        when(modelMapper.map(specialty, SpecialtyDTO.class)).thenReturn(specialtyDTO);

        SpecialtyDTO result = specialtyService.create(validRequest);

        assertThat(result).isNotNull();
        verify(specialtyRepository).save(specialty);
    }

    @Test
    @DisplayName("create: lanza RuntimeException si el nombre ya existe")
    void create_nombreDuplicado_lanzaExcepcion() {
        when(specialtyRepository.existsByName("Pediatría")).thenReturn(true);

        assertThatThrownBy(() -> specialtyService.create(validRequest))
                .isInstanceOf(RuntimeException.class)
                .hasMessageContaining("Specialty already exists");
    }

    // =========================================================================
    // update()
    // =========================================================================

    @Test
    @DisplayName("update: actualiza nombre y descripción de la especialidad")
    void update_especialidadExiste_actualizaDatos() {
        SpecialtyRequest updateReq = SpecialtyRequest.builder()
                .name("Pediatría General")
                .description("Medicina para niños y adolescentes")
                .build();

        when(specialtyRepository.findById(specialty.getId())).thenReturn(Optional.of(specialty));
        when(specialtyRepository.save(any())).thenReturn(specialty);
        when(modelMapper.map(eq(specialty), eq(SpecialtyDTO.class))).thenReturn(specialtyDTO);

        specialtyService.update(specialty.getId(), updateReq);

        assertThat(specialty.getName()).isEqualTo("Pediatría General");
        assertThat(specialty.getDescription()).isEqualTo("Medicina para niños y adolescentes");
        verify(specialtyRepository).save(specialty);
    }

    @Test
    @DisplayName("update: lanza ResourceNotFoundException si la especialidad no existe")
    void update_noExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(specialtyRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> specialtyService.update(fakeId, validRequest))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    // =========================================================================
    // delete() y activate()
    // =========================================================================

    @Test
    @DisplayName("delete: desactiva la especialidad (soft delete)")
    void delete_especialidadExiste_laDesactiva() {
        when(specialtyRepository.findById(specialty.getId())).thenReturn(Optional.of(specialty));

        specialtyService.delete(specialty.getId());

        // delete() es un soft delete — solo pone active=false, no borra la fila
        assertThat(specialty.isActive()).isFalse();
        verify(specialtyRepository).save(specialty);
    }

    @Test
    @DisplayName("activate: reactiva una especialidad desactivada")
    void activate_especialidadDesactivada_laActiva() {
        specialty.setActive(false);
        when(specialtyRepository.findById(specialty.getId())).thenReturn(Optional.of(specialty));

        specialtyService.activate(specialty.getId());

        assertThat(specialty.isActive()).isTrue();
        verify(specialtyRepository).save(specialty);
    }

    @Test
    @DisplayName("delete: lanza ResourceNotFoundException si la especialidad no existe")
    void delete_noExiste_lanzaExcepcion() {
        UUID fakeId = UUID.randomUUID();
        when(specialtyRepository.findById(fakeId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> specialtyService.delete(fakeId))
                .isInstanceOf(ResourceNotFoundException.class);
    }
}
