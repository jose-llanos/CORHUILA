package com.map.parking_project.services;

import com.map.parking_project.models.VehicleEntry;
import com.map.parking_project.repositories.IVehicleEntryRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class VehicleEntryServiceTest {

    @Mock
    private IVehicleEntryRepository repository;

    @InjectMocks
    private VehicleEntryService vehicleEntryService;

    private VehicleEntry entry;

    @BeforeEach
    void setUp() {
        entry = new VehicleEntry();
        entry.setId(1L);
        // Agrega setPlaca o lo que tenga tu modelo
    }

    @Test
    void testSave() {
        when(repository.save(any(VehicleEntry.class))).thenReturn(entry);
        VehicleEntry saved = vehicleEntryService.save(new VehicleEntry());
        assertNotNull(saved);
        assertEquals(1L, saved.getId());
    }

    @Test
    void testFindAll() {
        when(repository.findAll()).thenReturn(Arrays.asList(entry));
        List<VehicleEntry> result = vehicleEntryService.findAll();
        assertEquals(1, result.size());
    }

    @Test
    void testDelete() {
        doNothing().when(repository).deleteById(1L);
        vehicleEntryService.delete(1L);
        verify(repository, times(1)).deleteById(1L);
    }
}