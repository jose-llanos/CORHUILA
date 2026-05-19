package com.map.parking_project.services;

import com.map.parking_project.models.Tarifa;
import com.map.parking_project.models.VehicleEntry;
import com.map.parking_project.repositories.ITarifaRepository;
import com.map.parking_project.repositories.IVehicleEntryRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ServicesBasicTest {

    @Mock
    private ITarifaRepository tarifaRepo;
    @InjectMocks
    private TarifaServices tarifaService;

    @Mock
    private IVehicleEntryRepository vehicleRepo;
    @InjectMocks
    private VehicleEntryService vehicleService;

    @Test
    void testTarifaService() {
        Tarifa t = new Tarifa();
        when(tarifaRepo.findAll()).thenReturn(Arrays.asList(t));
        when(tarifaRepo.findById(1L)).thenReturn(Optional.of(t));
        when(tarifaRepo.save(any())).thenReturn(t);

        assertNotNull(tarifaService.findAll());
        assertNotNull(tarifaService.findById(1L));
        assertNotNull(tarifaService.save(t));

        tarifaService.delete(1L);
        verify(tarifaRepo, times(1)).deleteById(1L);
    }

    @Test
    void testVehicleEntryService() {
        VehicleEntry v = new VehicleEntry();
        when(vehicleRepo.findAll()).thenReturn(Arrays.asList(v));
        when(vehicleRepo.save(any())).thenReturn(v);

        assertNotNull(vehicleService.findAll());
        assertNotNull(vehicleService.save(v));

        vehicleService.delete(1L);
        verify(vehicleRepo, times(1)).deleteById(1L);
    }
}