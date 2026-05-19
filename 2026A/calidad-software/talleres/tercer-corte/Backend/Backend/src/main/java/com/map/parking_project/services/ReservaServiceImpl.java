package com.map.parking_project.services;

import java.util.List;
import java.util.Optional;

import com.map.parking_project.models.Reservas;
import com.map.parking_project.repositories.IReservasRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

//  1. IMPORTS AÑADIDOS PARA EL LOGGER
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Service
public class ReservaServiceImpl implements IReservaService {

    //  2. DECLARACIÓN DEL LOGGER (Asociado a esta clase específica)
    private static final Logger logger = LoggerFactory.getLogger(ReservaServiceImpl.class);

    @Autowired
    private IReservasRepository repository;

    @Override
    @Transactional(readOnly = true)
    public List<Reservas> findAll() {
        return (List<Reservas>) repository.findAll();
    }

    @Override
    @Transactional(readOnly = true)
    public Optional<Reservas> findById(Long id) {
        return repository.findById(id);
    }

    @Override
    @Transactional
    public Reservas save(Reservas reserva) {
        return repository.save(reserva);
    }

    @Override
    @Transactional
    public void update(Reservas reserva, Long id) {
        Optional<Reservas> reservaActual = repository.findById(id);

        if (reservaActual.isPresent()) {
            Reservas res = reservaActual.get();
            res.setTipo_vehiculo(reserva.getTipo_vehiculo());
            res.setTipo_servicio(reserva.getTipo_servicio());
            res.setHoras(reserva.getHoras()); 
            res.setFecha(reserva.getFecha()); 
            res.setPrecio(reserva.getPrecio());
            res.setConfirmada(reserva.isConfirmada()); 
            repository.save(res);
        } else {
            //  3. CORREGIDO: Reemplazado System.out por un logger estructurado
            // Usamos {} para pasarle el ID dinámicamente, que es una excelente práctica que SonarQube ama.
            logger.warn("Reserva no encontrada con ID: {}", id);
        }
    }

    @Override
    @Transactional
    public void delete(Long id) {
        repository.deleteById(id);
    }
}