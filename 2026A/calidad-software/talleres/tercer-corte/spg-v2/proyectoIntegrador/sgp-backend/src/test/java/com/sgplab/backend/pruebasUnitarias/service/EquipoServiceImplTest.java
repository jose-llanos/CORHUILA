package com.sgplab.backend.pruebasUnitarias.service;

import com.sgplab.backend.model.entity.Equipo;
import com.sgplab.backend.repository.IEquipoRepository;
import com.sgplab.backend.service.EquipoService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class EquipoServiceTest {

    @Mock
    private IEquipoRepository equipoRepository;

    @InjectMocks
    private EquipoService equipoService; // CORRECCIÓN: Clase concreta, no interfaz

    @Test
    @DisplayName("SQA: Intentar eliminar un equipo que no existe debe lanzar RuntimeException")
    void testEliminarEquipo_NoExiste() {
        // Arrange
        Long equipoId = 1L;
        when(equipoRepository.existsById(equipoId)).thenReturn(false);

        // Act & Assert
        assertThrows(RuntimeException.class, () -> {
            equipoService.eliminarEquipo(equipoId);
        });
    }
}