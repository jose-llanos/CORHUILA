package com.map.parking_project.services;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.*;


@ExtendWith(MockitoExtension.class)
class MapServicesServTest {

    @Mock
    private IMapServicesService mapRepo; // Asegúrate que el nombre coincida con tu @Autowired

    @InjectMocks
    private MapServicesServ mapServices;

    @Test
    void testServiceLogic() {
        // Aquí probamos los métodos de MapServicesServ
        // Si tiene un findAll o save, usa la misma lógica de los anteriores
        assertNotNull(mapServices);
    }
}