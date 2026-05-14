package com.sgplab.backend.pruebasDeIntegracion;

import com.sgplab.backend.model.entity.Equipo;
import com.sgplab.backend.model.entity.Prestamo;
import com.sgplab.backend.model.entity.Usuario;
import com.sgplab.backend.model.enums.EstadoPrestamo;
import com.sgplab.backend.repository.IEquipoRepository;
import com.sgplab.backend.repository.IUsuarioRepository;
import com.sgplab.backend.repository.IPrestamoRepository;
import com.sgplab.backend.service.*;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.transaction.annotation.Transactional;
import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@Transactional
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
public class SgpIntegrationTest {

    @Autowired
    private EquipoService equipoService;

    @Autowired
    private IEquipoRepository equipoRepository;

    @Autowired
    private IUsuarioRepository usuarioRepository;

    @Autowired
    private IPrestamoRepository prestamoRepository;

    @Test
    void debeValidarRestriccionDeEliminacion() {
        Usuario admin = usuarioRepository.findAll().stream()
                .filter(u -> "admin@sgplab.edu.co".equals(u.getEmail()))
                .findFirst()
                .orElseThrow(() -> new RuntimeException("Admin no encontrado"));

        Equipo equipo = equipoRepository.findAll().stream()
                .filter(e -> "CEN-002".equals(e.getCodigoInventario()))
                .findFirst()
                .orElseThrow(() -> new RuntimeException("Equipo CEN-002 no encontrado"));

        Prestamo prestamo = new Prestamo();
        prestamo.setUsuario(admin);
        prestamo.setEquipo(equipo);
        prestamo.setEstado(EstadoPrestamo.ACTIVO);
        prestamo.setFechaInicio(LocalDate.now());
        prestamo.setFechaFin(LocalDate.now().plusDays(2));

        prestamoRepository.save(prestamo);
        prestamoRepository.flush(); // Aseguramos que el préstamo exista en la DB

        assertThrows(RuntimeException.class, () -> {
            equipoService.eliminarEquipo(equipo.getId());
            equipoRepository.flush(); //
        });
    }

    @Test
    void debeVerificarDatosDePrimerEquipo() {
        Equipo equipo = equipoRepository.findAll().stream()
                .filter(e -> "CEN-002".equals(e.getCodigoInventario()))
                .findFirst()
                .orElse(null);

        assertNotNull(equipo, "La centrífuga debería existir en Docker");
        assertTrue(equipo.getNombre().contains("Centrífuga Digital"));
    }

    @Test
    void debeVerificarUsuarioAdmin() {
        Usuario usuario = usuarioRepository.findAll().stream()
                .filter(u -> "admin@sgplab.edu.co".equals(u.getEmail()))
                .findFirst()
                .orElse(null);

        assertNotNull(usuario, "El admin debería estar registrado");
        assertEquals("admin", usuario.getNombre());
    }
}