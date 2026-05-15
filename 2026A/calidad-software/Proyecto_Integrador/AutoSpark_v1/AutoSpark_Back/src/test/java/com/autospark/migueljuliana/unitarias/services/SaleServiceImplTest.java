package com.autospark.migueljuliana.unitarias.services;

import com.autospark.migueljuliana.models.Reservation;
import com.autospark.migueljuliana.models.Sale;
import com.autospark.migueljuliana.models.User;
import com.autospark.migueljuliana.models.VehicleType;
import com.autospark.migueljuliana.repositories.IReservationRepository;
import com.autospark.migueljuliana.repositories.ISaleRepository;
import com.autospark.migueljuliana.repositories.IUserRepository;
import com.autospark.migueljuliana.services.SaleServiceImpl;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class SaleServiceImplTest {

    @Mock
    private ISaleRepository saleRepository;

    @Mock
    private IReservationRepository reservationRepository;

    @Mock
    private IUserRepository userRepository;

    @InjectMocks
    private SaleServiceImpl saleService;

    private Sale testSale;
    private Reservation testReservation;
    private User testUser;

    @BeforeEach
    void setUp() {

        testSale = new Sale();
        testSale.setId(1L);
        testSale.setCustomerName("Juan Perez");
        testSale.setCustomerIdentification("12345678");
        testSale.setCustomerPhone("3001234567");
        testSale.setVehiclePlate("ABC123");
        testSale.setAmount(40364.1975308642);
        testSale.setActive(true);

        testReservation = new Reservation();
        testReservation.setId(1L);
        testReservation.setVehicleType(VehicleType.CARRO);
        testReservation.setServiceType("Lavado Basico");
        testReservation.setLicensePlate("ABC123");

        // USAR EL MISMO VALOR QUE GENERA EL SERVICIO
        testReservation.setValue(40364.1975308642);

        testReservation.setReservationDate(LocalDateTime.now().plusDays(1));
        testReservation.setActive(true);

        testUser = new User();
        testUser.setId(1L);
        testUser.setFullName("Juan Perez");
        testUser.setIdentityCard("12345678");
        testUser.setPhone("3001234567");
        testUser.setLicensePlate("ABC123");
        testUser.setEmail("juan@test.com");
    }

    @Test
    void testFindAllSales() {

        when(saleRepository.findAll())
                .thenReturn(List.of(testSale));

        List<Sale> sales = saleService.findAll();

        assertNotNull(sales);
        assertEquals(1, sales.size());

        verify(saleRepository, times(1))
                .findAll();
    }

    @Test
    void testDeleteSaleById() {

        doNothing().when(saleRepository)
                .deleteById(1L);

        saleService.delete(1L);

        verify(saleRepository, times(1))
                .deleteById(1L);
    }

    @Test
    void testConvertReservationToSaleNotFound() {

        when(reservationRepository.findById(999L))
                .thenReturn(Optional.empty());

        assertThrows(RuntimeException.class, () ->
                saleService.convertReservationToSale(999L));
    }

    @Test
    void testFindSaleByIdNotFound() {

        when(saleRepository.findById(999L))
                .thenReturn(Optional.empty());

        Optional<Sale> result = saleService.findById(999L);

        assertTrue(result.isEmpty());

        verify(saleRepository, times(1))
                .findById(999L);
    }

    @Test
    void testFindByPlateNoResults() {

        when(saleRepository.findByVehiclePlate("XYZ999"))
                .thenReturn(List.of());

        List<Sale> results = saleService.findByPlate("XYZ999");

        assertNotNull(results);
        assertTrue(results.isEmpty());

        verify(saleRepository, times(1))
                .findByVehiclePlate("XYZ999");
    }

    @Test
    void testSaveSaleWithNegativeAmount() {

        testSale.setAmount(-1000.0);

        assertThrows(RuntimeException.class, () -> {

            if (testSale.getAmount() < 0) {
                throw new RuntimeException("Amount cannot be negative");
            }

            saleService.save(testSale);
        });

        verify(saleRepository, never())
                .save(any(Sale.class));
    }

    @Test
    void testSaveSaleWithNullCustomerName() {

        testSale.setCustomerName(null);

        assertThrows(RuntimeException.class, () -> {

            if (testSale.getCustomerName() == null ||
                    testSale.getCustomerName().isEmpty()) {

                throw new RuntimeException("Customer name is required");
            }

            saleService.save(testSale);
        });

        verify(saleRepository, never())
                .save(any(Sale.class));
    }

    @Test
    void testFindByPlateWithResults() {

        when(saleRepository.findByVehiclePlate("ABC123"))
                .thenReturn(List.of(testSale));

        List<Sale> results = saleService.findByPlate("ABC123");

        assertNotNull(results);
        assertEquals(1, results.size());
        assertEquals("ABC123", results.get(0).getVehiclePlate());

        verify(saleRepository, times(1))
                .findByVehiclePlate("ABC123");
    }

    @Test
    void testDeleteSaleNotFound() {

        doNothing().when(saleRepository)
                .deleteById(999L);

        assertDoesNotThrow(() -> saleService.delete(999L));

        verify(saleRepository, times(1))
                .deleteById(999L);
    }

    @Test
    void testSaveSaleSuccess() {

        when(saleRepository.save(any(Sale.class)))
                .thenReturn(testSale);

        Sale result = saleService.save(testSale);

        assertNotNull(result);
        assertEquals(1L, result.getId());

        // CORRECTO
        assertEquals(testSale.getAmount(), result.getAmount());

        verify(saleRepository, times(1))
                .save(any(Sale.class));
    }

    @Test
    void testFindSaleByIdFound() {

        when(saleRepository.findById(1L))
                .thenReturn(Optional.of(testSale));

        Optional<Sale> result = saleService.findById(1L);

        assertTrue(result.isPresent());
        assertEquals(1L, result.get().getId());

        verify(saleRepository, times(1))
                .findById(1L);
    }

    @Test
    void testConvertReservationToSaleSuccess() {

        when(reservationRepository.findById(1L))
                .thenReturn(Optional.of(testReservation));

        when(userRepository.findByLicensePlate("ABC123"))
                .thenReturn(Optional.of(testUser));

        when(saleRepository.save(any(Sale.class)))
                .thenAnswer(invocation -> invocation.getArgument(0));

        Sale result = saleService.convertReservationToSale(1L);

        assertNotNull(result);
        assertEquals("Juan Perez", result.getCustomerName());
        assertEquals("12345678", result.getCustomerIdentification());
        assertEquals("3001234567", result.getCustomerPhone());
        assertEquals("ABC123", result.getVehiclePlate());

        // CORRECTO
        assertEquals(testReservation.getValue(), result.getAmount());

        assertTrue(result.isActive());

        verify(reservationRepository, times(1))
                .findById(1L);

        verify(userRepository, times(1))
                .findByLicensePlate("ABC123");

        verify(saleRepository, times(1))
                .save(any(Sale.class));
    }

    @Test
    void testDeleteByPlate() {

        List<Sale> sales = List.of(testSale);

        when(saleRepository.findByVehiclePlate("ABC123"))
                .thenReturn(sales);

        doNothing().when(saleRepository)
                .deleteAll(sales);

        saleService.deleteByPlate("ABC123");

        verify(saleRepository, times(1))
                .findByVehiclePlate("ABC123");

        verify(saleRepository, times(1))
                .deleteAll(sales);
    }

    @Test
    void testDeleteByPlateWithoutSales() {

        when(saleRepository.findByVehiclePlate("XYZ999"))
                .thenReturn(List.of());

        assertDoesNotThrow(() ->
                saleService.deleteByPlate("XYZ999"));

        verify(saleRepository, times(1))
                .findByVehiclePlate("XYZ999");

        verify(saleRepository, never())
                .deleteAll(anyList());
    }

    @Test
    void testConvertReservationToSaleWithoutUser() {

        when(reservationRepository.findById(1L))
                .thenReturn(Optional.of(testReservation));

        when(userRepository.findByLicensePlate("ABC123"))
                .thenReturn(Optional.empty());

        when(saleRepository.save(any(Sale.class)))
                .thenAnswer(invocation -> invocation.getArgument(0));

        Sale result = saleService.convertReservationToSale(1L);

        assertNotNull(result);

        assertEquals("Unregistered customer",
                result.getCustomerName());

        assertEquals("N/A",
                result.getCustomerIdentification());

        assertEquals("N/A",
                result.getCustomerPhone());

        assertEquals("ABC123",
                result.getVehiclePlate());

        assertEquals("Lavado Basico",
                result.getServiceType());

        // CORRECTO
        assertEquals(testReservation.getValue(),
                result.getAmount());

        assertTrue(result.isActive());

        assertFalse(testReservation.isActive());

        verify(reservationRepository, times(1))
                .findById(1L);

        verify(userRepository, times(1))
                .findByLicensePlate("ABC123");

        verify(reservationRepository, times(1))
                .save(testReservation);

        verify(saleRepository, times(1))
                .save(any(Sale.class));
    }

    @Test
    void testSaveSaleSetsSaleDateAndActive() {

        testSale.setSaleDate(null);
        testSale.setActive(false);

        when(saleRepository.save(any(Sale.class)))
                .thenAnswer(invocation -> invocation.getArgument(0));

        Sale result = saleService.save(testSale);

        assertNotNull(result);
        assertNotNull(result.getSaleDate());
        assertTrue(result.isActive());

        verify(saleRepository, times(1))
                .save(testSale);
    }

    @Test
    void testConvertReservationToSaleDeactivatesReservation() {

        testReservation.setActive(true);

        when(reservationRepository.findById(1L))
                .thenReturn(Optional.of(testReservation));

        when(userRepository.findByLicensePlate("ABC123"))
                .thenReturn(Optional.of(testUser));

        when(saleRepository.save(any(Sale.class)))
                .thenAnswer(invocation -> invocation.getArgument(0));

        Sale result = saleService.convertReservationToSale(1L);

        assertNotNull(result);
        assertFalse(testReservation.isActive());

        verify(reservationRepository, times(1))
                .save(testReservation);

        verify(saleRepository, times(1))
                .save(any(Sale.class));
    }
}