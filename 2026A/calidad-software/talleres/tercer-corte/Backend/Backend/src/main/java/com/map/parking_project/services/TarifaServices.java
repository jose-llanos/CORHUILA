package com.map.parking_project.services;

import com.map.parking_project.models.Tarifa;
import com.map.parking_project.repositories.ITarifaRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;


@Service
public class TarifaServices implements ITarifaServices {

    @Autowired
    private ITarifaRepository tarifaDao;

    @Override
    public List<Tarifa> findAll() {
        return (List<Tarifa>) tarifaDao.findAll();
    }

    @Override
    public Tarifa findById(Long id) {
        return tarifaDao.findById(id).orElse(null);
    }

    @Override
    public Tarifa save(Tarifa tarifa) {
        return tarifaDao.save(tarifa);
    }

    @Override
    public void delete(Long id) {
        tarifaDao.deleteById(id);
    }
}
