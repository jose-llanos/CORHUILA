package com.sgplab.backend.pruebasUnitarias.service;

import com.sgplab.backend.repository.IPenalizacionRepository;
import com.sgplab.backend.service.PenalizacionService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.verify;

@ExtendWith(MockitoExtension.class)
public class PenalizacionServiceTest {

    @Mock
    private IPenalizacionRepository penalizacionRepository;

    @InjectMocks
    private PenalizacionService penalizacionService;

    @Test
    @DisplayName("SQA Baseline: Verificar lógica de detección de penalizado")
    void testIsUsuarioPenalizado() {
        // Valida que el servicio identifique correctamente a un usuario con sanciones activas
        Long id = 1L;
        Mockito.when(penalizacionRepository.existsByUsuarioId(id)).thenReturn(true);

        boolean estaPenalizado = penalizacionService.isUsuarioPenalizado(id);

        assertTrue(estaPenalizado);
        verify(penalizacionRepository).existsByUsuarioId(id);
    }
}