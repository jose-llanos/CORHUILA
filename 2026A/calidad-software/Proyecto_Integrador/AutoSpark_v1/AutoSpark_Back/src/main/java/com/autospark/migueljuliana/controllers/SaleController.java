package com.autospark.migueljuliana.controllers;

import com.autospark.migueljuliana.models.Sale;
import com.autospark.migueljuliana.services.ISaleService;
import com.autospark.migueljuliana.exception.ResourceNotFoundException;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Slf4j
@CrossOrigin(origins = "http://localhost:4200", allowCredentials = "true")
@RestController
@RequestMapping("/autospark/sales")
@RequiredArgsConstructor
public class SaleController {

    // ERROR LOW: Logger no se usa en algunos métodos
    private final ISaleService saleService;

    @GetMapping
    public ResponseEntity<List<Sale>> getAllSales() {
        // No usa log.debug ni log.info
        List<Sale> sales = saleService.findAll();
        return ResponseEntity.ok(sales);
    }

    @GetMapping("/{id}")
    public ResponseEntity<Sale> getSaleById(@PathVariable Long id) {
        log.debug("Fetching sale with id: {}", id);

        Sale sale = saleService.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Sale", id));

        return ResponseEntity.ok(sale);
    }

    @PostMapping
    public ResponseEntity<Sale> createSale(@RequestBody Sale sale) {
        log.info("Creating new sale for vehicle: {}", sale.getVehiclePlate());

        Sale newSale = saleService.save(sale);

        log.info("Sale created successfully with id: {}", newSale.getId());

        return ResponseEntity.status(HttpStatus.CREATED).body(newSale);
    }

    @PostMapping("/convert/{reservationId}")
    public ResponseEntity<Sale> convertReservationToSale(@PathVariable Long reservationId) {
        log.info("Converting reservation with id: {} to sale", reservationId);

        Sale sale = saleService.convertReservationToSale(reservationId);

        log.info("Reservation {} converted to sale successfully with id: {}", reservationId, sale.getId());

        return ResponseEntity.ok(sale);
    }

    @DeleteMapping("/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void deleteSale(@PathVariable Long id) {
        log.info("Deleting sale with id: {}", id);

        saleService.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Sale", id));

        saleService.delete(id);

        log.info("Sale with id: {} deleted successfully", id);
    }

    @DeleteMapping("/by-plate/{plate}")
    public ResponseEntity<Void> deleteSalesByPlate(@PathVariable String plate) {
        log.warn("Deleting all sales for vehicle plate: {}", plate);

        saleService.deleteByPlate(plate);

        return ResponseEntity.ok().build();
    }
}