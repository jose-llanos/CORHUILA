package com.map.parking_project.controllers;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import com.map.parking_project.models.Reservas;
import com.map.parking_project.services.IReservaService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import com.map.parking_project.dto.ReservaDTO;


@CrossOrigin(origins = { "http://localhost:4200" })
@RestController
@RequestMapping("/api")
public class ReservaRestController {

@Autowired
    private IReservaService reservaService;

    // 1. CONSTANTES AÑADIDAS PARA EVITAR DUPLICIDAD DE STRINGS (Code Smells resueltos)
    private static final String MESSAGE_KEY = "message";
    private static final String RESERVA_NO_ENCONTRADA_MSG = "Reserva no encontrada";

    // 2. CONSTRUCTOR VACÍO ELIMINADO 
    // Spring Boot inyecta @Autowired automáticamente, no necesitamos el constructor vacío.

    @GetMapping("/reservas")
    public List<Reservas> listarReservas() {
        return reservaService.findAll();
    }

    // 3. CAMBIO DE <?> a <Object>
    @GetMapping("/reservas/{id}")
    public ResponseEntity<Object> obtenerReserva(@PathVariable Long id) {
        Optional<Reservas> reserva = reservaService.findById(id);

        if (reserva.isPresent()) {
            return ResponseEntity.ok(reserva.get());
        } else {
            // Estandarizado a JSON (Map) y usando la constante
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(Map.of(MESSAGE_KEY, RESERVA_NO_ENCONTRADA_MSG));
        }
    }

    // 4. CAMBIO DE <?> a <Map<String, Object>> y uso de constante MESSAGE_KEY
    /* ¡CORRECCIÓN SONARQUBE!
     * Se cambió ResponseEntity<?> a ResponseEntity<Map<String, Object>>
     * para retornar un JSON con el mensaje de éxito o error.
     */
    /*
     * Registra una nueva reserva en el sistema.
     * @param reservaDTO Objeto con los datos de la reserva.
     * @return ResponseEntity con el mensaje de éxito o error.
     */
    @PostMapping("/reservas")
    public ResponseEntity<Map<String, Object>> registrarReserva(@RequestBody ReservaDTO reservaDTO) {
        try {
            Reservas reservaEntidad = new Reservas();
            
            // ✅ Mapeo correcto con los métodos reales de tu clase
            reservaEntidad.setTipo_vehiculo(reservaDTO.getTipo_vehiculo());
            reservaEntidad.setTipo_servicio(reservaDTO.getTipo_servicio());
            reservaEntidad.setHoras(reservaDTO.getHoras());
            reservaEntidad.setFecha(reservaDTO.getFecha());
            reservaEntidad.setPrecio(reservaDTO.getPrecio());
            
            // Configuraciones de seguridad por defecto
            reservaEntidad.setId(null); 
            reservaEntidad.setConfirmada(false);

            Reservas nuevaReserva = reservaService.save(reservaEntidad);

            return ResponseEntity.status(HttpStatus.CREATED).body(Map.of(
                    MESSAGE_KEY, "Reserva registrada correctamente",
                    "reserva", nuevaReserva
            ));

        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(Map.of(
                    MESSAGE_KEY, "Error al registrar la reserva",
                    "error", e.getMessage()
            ));
        }
    }

    @PutMapping("/reservas/{id}")
    public ResponseEntity<Map<String, Object>> actualizarReserva(@RequestBody ReservaDTO reservaDTO, @PathVariable Long id) {
        Optional<Reservas> reservaActualOpt = reservaService.findById(id);

        if (reservaActualOpt.isPresent()) {
            Reservas reservaActual = reservaActualOpt.get();

            // ✅ Mapeo correcto con los métodos reales de tu clase
            reservaActual.setTipo_vehiculo(reservaDTO.getTipo_vehiculo());
            reservaActual.setTipo_servicio(reservaDTO.getTipo_servicio());
            reservaActual.setHoras(reservaDTO.getHoras());
            reservaActual.setFecha(reservaDTO.getFecha());
            reservaActual.setPrecio(reservaDTO.getPrecio());

            reservaService.update(reservaActual, id);

            return ResponseEntity.ok(Map.of(
                    MESSAGE_KEY, "Reserva actualizada correctamente",
                    "reserva", reservaActual
            ));
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(Map.of(
                    MESSAGE_KEY, "Reserva no fue encontrada"
            ));
        }
    }

    @PutMapping("/reservas/{id}/confirmar")
    public ResponseEntity<Map<String, String>> confirmarReserva(@PathVariable Long id) {
        Optional<Reservas> reserva = reservaService.findById(id);

        if (reserva.isPresent()) {
            Reservas reservaActual = reserva.get();
            reservaActual.setConfirmada(true);
            reservaService.save(reservaActual);
            // Estandarizado a JSON
            return ResponseEntity.ok(Map.of(MESSAGE_KEY, "Reserva confirmada correctamente"));
        } else {
            // Uso de la constante y estandarizado a JSON
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(Map.of(MESSAGE_KEY, RESERVA_NO_ENCONTRADA_MSG));
        }
    }
    /* ¡CORRECCIÓN SONARQUBE!
     * Se cambió ResponseEntity<?> a ResponseEntity<Map<String, String>>
     * para retornar un JSON con el mensaje de éxito o error.
     */
    /*
     * Elimina una reserva del sistema.
     *@param id Identificador de la reserva a eliminar.
     *@return ResponseEntity indicando si la eliminación fue exitosa o si no se encontró.
     */
    @DeleteMapping("/reservas/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public ResponseEntity<Map<String, String>> eliminarReserva(@PathVariable Long id) {
        Optional<Reservas> reserva = reservaService.findById(id);

        if (reserva.isPresent()) {
            reservaService.delete(id);
            return ResponseEntity.ok(Map.of(MESSAGE_KEY, "Reserva eliminada correctamente"));
        } else {
            // Corrección lógica: Ahora arroja error 404 si no encuentra la reserva a borrar
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(Map.of(MESSAGE_KEY, RESERVA_NO_ENCONTRADA_MSG));
        }
    }


}

