package com.map.parking_project.services;

import com.map.parking_project.models.Tarifa;

import java.util.List;

public interface ITarifaServices {

    List<Tarifa> findAll();
    Tarifa findById(Long id);
    Tarifa save(Tarifa tarifa);
    void delete(Long id);
}
