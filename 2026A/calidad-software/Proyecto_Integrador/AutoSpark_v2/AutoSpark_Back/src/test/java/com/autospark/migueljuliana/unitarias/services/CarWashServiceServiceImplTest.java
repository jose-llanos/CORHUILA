package com.autospark.migueljuliana.unitarias.services;

import com.autospark.migueljuliana.exception.ResourceNotFoundException;
import com.autospark.migueljuliana.models.CarWashService;
import com.autospark.migueljuliana.repositories.ICarWashServiceRepository;
import com.autospark.migueljuliana.services.CarWashServiceServiceImpl;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class CarWashServiceServiceImplTest {

    @Mock
    private ICarWashServiceRepository repository;

    @InjectMocks
    private CarWashServiceServiceImpl service;

    private CarWashService testService;

    @BeforeEach
    void setUp() {
        testService = new CarWashService();
        testService.setId(1L);
        testService.setName("Lavado Basico");
        testService.setDescription("Lavado exterior e interior");
        testService.setPrice(25000.0);
        testService.setActive(true);
        testService.setImageUrl("http://example.com/image.jpg");
    }

    @Test
    void testCreateService() {
        when(repository.save(any(CarWashService.class))).thenReturn(testService);

        CarWashService saved = service.save(testService);

        assertNotNull(saved);
        assertEquals("Lavado Basico", saved.getName());
        assertEquals(25000.0, saved.getPrice());
        verify(repository, times(1)).save(any(CarWashService.class));
    }

    @Test
    void testFindAllServices() {
        when(repository.findAll()).thenReturn(java.util.List.of(testService));

        var services = service.findAll();

        assertNotNull(services);
        assertEquals(1, services.size());
        verify(repository, times(1)).findAll();
    }

    @Test
    void testUpdateServiceNotFound() {
        when(repository.findById(999L)).thenReturn(Optional.empty());

        assertThrows(ResourceNotFoundException.class, () -> service.update(testService, 999L));

        verify(repository, never()).save(any(CarWashService.class));
    }

    @Test
    void testDeleteServiceNotFound() {
        doNothing().when(repository).deleteById(999L);

        assertDoesNotThrow(() -> service.delete(999L));

        verify(repository, times(1)).deleteById(999L);
    }

    @Test
    void testFindServiceByIdNotFound() {
        when(repository.findById(999L)).thenReturn(Optional.empty());

        Optional<CarWashService> result = service.findById(999L);

        assertTrue(result.isEmpty());
        verify(repository, times(1)).findById(999L);
    }

    @Test
    void testUpdateServiceNotFoundError() {
        when(repository.findById(999L)).thenReturn(Optional.empty());

        ResourceNotFoundException exception = assertThrows(
                ResourceNotFoundException.class,
                () -> service.update(testService, 999L)
        );

        assertEquals("Service with id 999 not found", exception.getMessage());
        verify(repository, never()).save(any(CarWashService.class));
    }

    @Test
    void testActivateService() {
        testService.setActive(false);
        when(repository.findById(1L)).thenReturn(Optional.of(testService));
        when(repository.save(any(CarWashService.class))).thenReturn(testService);

        testService.setActive(true);
        service.update(testService, 1L);

        assertTrue(testService.isActive());
        verify(repository, times(1)).save(testService);
    }

    @Test
    void testDeactivateService() {
        testService.setActive(true);
        when(repository.findById(1L)).thenReturn(Optional.of(testService));
        when(repository.save(any(CarWashService.class))).thenReturn(testService);

        testService.setActive(false);
        service.update(testService, 1L);

        assertFalse(testService.isActive());
        verify(repository, times(1)).save(testService);
    }
}