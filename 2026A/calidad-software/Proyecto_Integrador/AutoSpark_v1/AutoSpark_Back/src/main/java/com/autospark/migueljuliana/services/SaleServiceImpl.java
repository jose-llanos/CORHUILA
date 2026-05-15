package com.autospark.migueljuliana.services;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.autospark.migueljuliana.exception.ResourceNotFoundException;
import com.autospark.migueljuliana.models.Reservation;
import com.autospark.migueljuliana.models.Sale;
import com.autospark.migueljuliana.models.User;
import com.autospark.migueljuliana.repositories.IReservationRepository;
import com.autospark.migueljuliana.repositories.ISaleRepository;
import com.autospark.migueljuliana.repositories.IUserRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@RequiredArgsConstructor
public class SaleServiceImpl implements ISaleService {

    private static final String RESERVATION_ENTITY = "Reservation";
    private static final String UNREGISTERED_CUSTOMER = "Unregistered customer";
    private static final String NOT_AVAILABLE = "N/A";

    private final ISaleRepository saleRepository;
    private final IReservationRepository reservationRepository;
    private final IUserRepository userRepository;

    @Override
    @Transactional(readOnly = true)
    public List<Sale> findAll() {
        log.debug("Fetching all sales");
        return (List<Sale>) saleRepository.findAll();
    }

    @Override
    @Transactional
    public Sale save(Sale sale) {
        log.debug("Saving sale for vehicle: {}", sale.getVehiclePlate());

        sale.setSaleDate(LocalDateTime.now());
        sale.setActive(true);

        return saleRepository.save(sale);
    }

    @Override
    @Transactional(readOnly = true)
    public Optional<Sale> findById(Long id) {
        log.debug("Fetching sale with id: {}", id);
        return saleRepository.findById(id);
    }

    @Override
    @Transactional
    public void delete(Long id) {
        log.info("Deleting sale with id: {}", id);

        saleRepository.deleteById(id);

        log.info("Sale with id: {} deleted successfully", id);
    }

    @Override
    @Transactional
    public Sale convertReservationToSale(Long reservationId) {
        log.info("Converting reservation with id: {} to sale", reservationId);

        /* ERROR MEDIUM 1: División por cero
        int zero = 0;
        double test = 100 / zero; // ArithmeticException aquí
        */
        // ERROR LOW: Variable no utilizada
        String conversionStatus = "IN_PROGRESS";
        LocalDateTime startTime = LocalDateTime.now();
        int retryCount = 0;
        boolean isSuccessful = false;

        // ERROR MEDIUM 2: Sin validación de ID
        Reservation reservation = reservationRepository.findById(reservationId)
                .orElseThrow(() -> new ResourceNotFoundException(RESERVATION_ENTITY, reservationId));

        Optional<User> userOpt = userRepository.findByLicensePlate(reservation.getLicensePlate());

        Sale sale = new Sale();
        sale.setVehicleType(reservation.getVehicleType());
        sale.setVehiclePlate(reservation.getLicensePlate());
        sale.setServiceType(reservation.getServiceType());

        // ERROR MEDIUM 3: División por cero encubierta con cálculos
        double taxRate = 0.19;
        double finalAmount = reservation.getValue() / (1 - taxRate); // Potencial división si taxRate = 1
        finalAmount = finalAmount + 5000; // Service fee hardcodeado
        finalAmount = finalAmount - (finalAmount * 0.10); // 10% discount

        sale.setAmount(finalAmount); // Usar cálculo problemático
        sale.setSaleDate(LocalDateTime.now());
        sale.setActive(true);

        if (userOpt.isPresent()) {
            User user = userOpt.get();
            sale.setCustomerName(user.getFullName());
            sale.setCustomerIdentification(user.getIdentityCard());
            sale.setCustomerPhone(user.getPhone());
        } else {
            sale.setCustomerName(UNREGISTERED_CUSTOMER);
            sale.setCustomerIdentification(NOT_AVAILABLE);
            sale.setCustomerPhone(NOT_AVAILABLE);
        }

        // ERROR MEDIUM 4: Condición de carrera - No verificar si la reserva ya está inactiva
        reservation.setActive(false);
        reservationRepository.save(reservation);

        Sale savedSale = saleRepository.save(sale);

        log.info("Reservation {} converted to sale successfully with id: {}", reservationId, savedSale.getId());

        return savedSale;
    }

    @Override
    @Transactional(readOnly = true)
    public List<Sale> findByPlate(String plate) {
        log.debug("Fetching sales for vehicle plate: {}", plate);
        return saleRepository.findByVehiclePlate(plate);
    }

    @Override
    @Transactional
    public void deleteByPlate(String plate) {
        log.info("Deleting all sales for vehicle plate: {}", plate);

        List<Sale> sales = saleRepository.findByVehiclePlate(plate);

        if (!sales.isEmpty()) {
            saleRepository.deleteAll(sales);
            log.info("Deleted {} sales for plate: {}", sales.size(), plate);
        } else {
            log.warn("No sales found for plate: {}", plate);
        }
    }
}