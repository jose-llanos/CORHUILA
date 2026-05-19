package com.autospark.migueljuliana.unitarias.controllers;

import com.autospark.migueljuliana.controllers.ServiceController;
import com.autospark.migueljuliana.models.CarWashService;
import com.autospark.migueljuliana.services.ICarWashServiceService;
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
        controllers = ServiceController.class,
        excludeAutoConfiguration = {
                SecurityAutoConfiguration.class,
                SecurityFilterAutoConfiguration.class,
                OAuth2ResourceServerAutoConfiguration.class
        }
)
@AutoConfigureMockMvc(addFilters = false)
class ServiceControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private ICarWashServiceService service;

    @MockitoBean
    private JpaMetamodelMappingContext jpaMetamodelMappingContext;

    @MockitoBean(name = "auditorAwareImpl")
    private AuditorAware<String> auditorAware;

    private CarWashService crearServicio() {
        CarWashService carWashService = new CarWashService();
        carWashService.setId(1L);
        carWashService.setName("Lavado Basico");
        carWashService.setDescription("Lavado exterior del vehiculo");
        carWashService.setPrice(25000.0);
        carWashService.setActive(true);
        carWashService.setImageUrl("imagen.jpg");
        return carWashService;
    }

    @Test
    void getAllServicesDebeRetornarOk() throws Exception {
        when(service.findAll()).thenReturn(List.of(crearServicio()));

        mockMvc.perform(get("/autospark/service"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].id").value(1))
                .andExpect(jsonPath("$[0].name").value("Lavado Basico"))
                .andExpect(jsonPath("$[0].description").value("Lavado exterior del vehiculo"))
                .andExpect(jsonPath("$[0].price").value(25000.0))
                .andExpect(jsonPath("$[0].active").value(true));

        verify(service).findAll();
    }

    @Test
    void getServiceByIdDebeRetornarOk() throws Exception {
        when(service.findById(1L)).thenReturn(Optional.of(crearServicio()));

        mockMvc.perform(get("/autospark/service/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.name").value("Lavado Basico"))
                .andExpect(jsonPath("$.price").value(25000.0));

        verify(service).findById(1L);
    }

    @Test
    void createServiceDebeRetornarCreated() throws Exception {
        when(service.save(any(CarWashService.class))).thenReturn(crearServicio());

        mockMvc.perform(post("/autospark/service")
                        .contentType("application/json")
                        .content("""
                                {
                                  "name": "Lavado Basico",
                                  "description": "Lavado exterior del vehiculo",
                                  "price": 25000.0,
                                  "active": true,
                                  "imageUrl": "imagen.jpg"
                                }
                                """))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").value(1))
                .andExpect(jsonPath("$.name").value("Lavado Basico"))
                .andExpect(jsonPath("$.price").value(25000.0))
                .andExpect(jsonPath("$.active").value(true))
                .andExpect(jsonPath("$.imageUrl").value("imagen.jpg"));

        verify(service).save(any(CarWashService.class));
    }

    @Test
    void updateServiceDebeRetornarOk() throws Exception {
        CarWashService existing = crearServicio();

        CarWashService updated = crearServicio();
        updated.setName("Lavado Premium");
        updated.setDescription("Lavado completo interior y exterior");
        updated.setPrice(50000.0);
        updated.setImageUrl("premium.jpg");

        when(service.findById(1L)).thenReturn(Optional.of(existing));
        when(service.save(any(CarWashService.class))).thenReturn(updated);

        mockMvc.perform(put("/autospark/service/1")
                        .contentType("application/json")
                        .content("""
                                {
                                  "name": "Lavado Premium",
                                  "description": "Lavado completo interior y exterior",
                                  "price": 50000.0,
                                  "active": true,
                                  "imageUrl": "premium.jpg"
                                }
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.name").value("Lavado Premium"))
                .andExpect(jsonPath("$.description").value("Lavado completo interior y exterior"))
                .andExpect(jsonPath("$.price").value(50000.0))
                .andExpect(jsonPath("$.imageUrl").value("premium.jpg"));

        verify(service).findById(1L);
        verify(service).save(any(CarWashService.class));
    }

    @Test
    void deleteServiceDebeRetornarNoContent() throws Exception {
        when(service.findById(1L)).thenReturn(Optional.of(crearServicio()));
        doNothing().when(service).delete(1L);

        mockMvc.perform(delete("/autospark/service/1"))
                .andExpect(status().isNoContent());

        verify(service).findById(1L);
        verify(service).delete(1L);
    }
}