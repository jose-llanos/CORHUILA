package com.sgplab.backend.pruebasUnitarias.service;

import com.sgplab.backend.model.entity.Equipo;
import com.sgplab.backend.model.entity.Prestamo;
import com.sgplab.backend.model.entity.Usuario;
import com.sgplab.backend.model.enums.EstadoPrestamo;
import com.sgplab.backend.repository.IEquipoRepository;
import com.sgplab.backend.repository.IPrestamoRepository;
import com.sgplab.backend.service.PrestamoService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
public class PrestamoServiceTest {

    @Mock
    private IPrestamoRepository prestamoRepository;

    @Mock
    private IEquipoRepository equipoRepository;

    @InjectMocks
    private PrestamoService prestamoService;

    @Test
    @DisplayName("SQA: Verificar flujo básico de guardado de préstamo")
    void testCrearPrestamo_FlujoNormal() {
        // Valida que el servicio cree un préstamo exitosamente asegurando que: el usuario
        // no tenga préstamos activos, exista stock del equipo, se descuente dicho stock
        // y se asigne el estado ACTIVO al nuevo préstamo.

        // 1. Arrange (Preparar datos)
        Usuario usuario = new Usuario();
        usuario.setId(1L);

        Equipo equipo = new Equipo();
        equipo.setId(1L);
        equipo.setCantidad(5); // Le damos stock > 0 para superar la validación

        Prestamo prestamo = new Prestamo();
        prestamo.setUsuario(usuario);
        prestamo.setEquipo(equipo);

        // Simulamos que el usuario NO tiene préstamos activos
        when(prestamoRepository.existsByUsuarioIdAndEstado(1L, EstadoPrestamo.ACTIVO))
                .thenReturn(false);

        // Simulamos que la base de datos encuentra el equipo
        when(equipoRepository.findById(1L))
                .thenReturn(Optional.of(equipo));

        // Simulamos el guardado final del préstamo
        when(prestamoRepository.save(any(Prestamo.class)))
                .thenReturn(prestamo);

        // 2. Act (Ejecutar el método)
        Prestamo resultado = prestamoService.crearPrestamo(prestamo);

        // 3. Assert (Verificar los resultados y comportamientos)
        assertNotNull(resultado);
        assertEquals(EstadoPrestamo.ACTIVO, resultado.getEstado(), "El estado debe ser ACTIVO");
        assertEquals(4, equipo.getCantidad(), "El stock del equipo debió descontarse en 1");

        // Verificamos que se ejecutaron las llamadas a los repositorios
        verify(prestamoRepository).existsByUsuarioIdAndEstado(1L, EstadoPrestamo.ACTIVO);
        verify(equipoRepository).findById(1L);
        verify(equipoRepository).save(equipo); // Verifica que se guardó la actualización del stock
        verify(prestamoRepository).save(prestamo);
    }
}