package com.autospark.migueljuliana.unitarias.controllers;

import com.autospark.migueljuliana.controllers.SaleController;
import com.autospark.migueljuliana.models.Sale;
import com.autospark.migueljuliana.services.ISaleService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.security.oauth2.resource.servlet.OAuth2ResourceServerAutoConfiguration;
import org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration;
import org.springframework.boot.autoconfigure.security.servlet.SecurityFilterAutoConfiguration;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.data.domain.AuditorAware;
import org.springframework.data.jpa.mapping.JpaMetamodelMappingContext;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import java.util.List;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(
        controllers = SaleController.class,
        excludeAutoConfiguration = {
                SecurityAutoConfiguration.class,
                SecurityFilterAutoConfiguration.class,
                OAuth2ResourceServerAutoConfiguration.class
        }
)
@AutoConfigureMockMvc(addFilters = false)
class SaleControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private ISaleService saleService;

    @MockitoBean
    private JpaMetamodelMappingContext jpaMetamodelMappingContext;

    @MockitoBean(name = "auditorAwareImpl")
    private AuditorAware<String> auditorAware;

    private Sale crearVenta() {
        Sale sale = new Sale();
        sale.setId(1L);
        sale.setVehiclePlate("ABC123");
        sale.setAmount(25000.0);
        sale.setActive(true);
        return sale;
    }

    @Test
    void getAllSalesDebeRetornarOk() throws Exception {
        when(saleService.findAll()).thenReturn(List.of(crearVenta()));

        mockMvc.perform(get("/autospark/sales"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].id").value(1))
                .andExpect(jsonPath("$[0].vehiclePlate").value("ABC123"))
                .andExpect(jsonPath("$[0].amount").value(25000.0))
                .andExpect(jsonPath("$[0].active").value(true));

        verify(saleService).findAll();
    }

    @Test
    void getSaleByIdDebeRetornarOk() throws Exception {
        when(saleService.findById(1L)).thenReturn(Optional.of(crearVenta()));

        mockMvc.perform(get("/autospark/sales/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.vehiclePlate").value("ABC123"))
                .andExpect(jsonPath("$.amount").value(25000.0));

        verify(saleService).findById(1L);
    }

    @Test
    void createSaleDebeRetornarCreated() throws Exception {
        when(saleService.save(any(Sale.class))).thenReturn(crearVenta());

        mockMvc.perform(post("/autospark/sales")
                        .contentType("application/json")
                        .content("""
                                {
                                  "vehiclePlate": "ABC123",
                                  "amount": 25000.0,
                                  "active": true
                                }
                                """))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.vehiclePlate").value("ABC123"))
                .andExpect(jsonPath("$.amount").value(25000.0))
                .andExpect(jsonPath("$.active").value(true));

        verify(saleService).save(any(Sale.class));
    }

    @Test
    void convertReservationToSaleDebeRetornarOk() throws Exception {
        when(saleService.convertReservationToSale(1L)).thenReturn(crearVenta());

        mockMvc.perform(post("/autospark/sales/convert/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.vehiclePlate").value("ABC123"))
                .andExpect(jsonPath("$.amount").value(25000.0))
                .andExpect(jsonPath("$.active").value(true));

        verify(saleService).convertReservationToSale(1L);
    }

    @Test
    void deleteSaleDebeRetornarNoContent() throws Exception {
        when(saleService.findById(1L)).thenReturn(Optional.of(crearVenta()));
        doNothing().when(saleService).delete(1L);

        mockMvc.perform(delete("/autospark/sales/1"))
                .andExpect(status().isNoContent());

        verify(saleService).findById(1L);
        verify(saleService).delete(1L);
    }

    @Test
    void deleteSalesByPlateDebeRetornarOk() throws Exception {
        doNothing().when(saleService).deleteByPlate("ABC123");

        mockMvc.perform(delete("/autospark/sales/by-plate/ABC123"))
                .andExpect(status().isOk());

        verify(saleService).deleteByPlate("ABC123");
    }
}