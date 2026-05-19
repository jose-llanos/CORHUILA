package com.map.parking_project.services;

import com.map.parking_project.models.Reservas;

import java.util.List;
import java.util.Optional;

public interface IReservaService {

    public List<Reservas> findAll();

    public Optional<Reservas> findById(Long id);

    public Reservas save(Reservas reserva);

    public void update(Reservas reserva, Long id);

    public void delete(Long id);

}