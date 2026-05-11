package com.corhuila.gestionpruebas.selenium.tests;

import com.corhuila.gestionpruebas.selenium.pages.CitasPage;
import com.corhuila.gestionpruebas.selenium.pages.DuenosPage;
import com.corhuila.gestionpruebas.selenium.pages.MascotasPage;
import com.corhuila.gestionpruebas.selenium.utils.DataGenerator;

import org.junit.jupiter.api.MethodOrderer.OrderAnnotation;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

import static org.junit.jupiter.api.Assertions.assertTrue;

@TestMethodOrder(OrderAnnotation.class)
public class VeterinariaTest extends BaseTest {

    @Test
    @Order(1)
    void caso1_crearNuevoDueno() {
        driver.get(baseUrl + "/duenios");
        DuenosPage duenosPage = new DuenosPage(driver);
        String nombre = DataGenerator.generarNombre();
        duenosPage.irANuevoDueno();
        duenosPage.crearDueno(
                nombre,
                DataGenerator.generarTelefono(),
                DataGenerator.generarEmail()
        );
        assertTrue(duenosPage.existeDuenoEnTabla(nombre));
    }

    @Test
    @Order(2)
    void caso2_crearNuevaMascota() {
        // Primero crear un dueño para asegurar que haya al menos uno
        driver.get(baseUrl + "/duenios");
        DuenosPage duenosPage = new DuenosPage(driver);
        duenosPage.irANuevoDueno();
        duenosPage.crearDueno(
                DataGenerator.generarNombre(),
                DataGenerator.generarTelefono(),
                DataGenerator.generarEmail()
        );

        // Ahora crear la mascota
        driver.get(baseUrl + "/mascotas");
        MascotasPage mascotasPage = new MascotasPage(driver);
        String mascotaNombre = DataGenerator.generarNombreMascota();
        mascotasPage.irANuevaMascota();
        mascotasPage.crearMascota(
                mascotaNombre,
                "Perro",
                "Labrador",
                "3"
        );
        assertTrue(driver.getPageSource().contains(mascotaNombre));
    }

    @Test
    @Order(3)
    void caso3_validarCamposRequeridos() {
        driver.get(baseUrl + "/duenios/nuevo");
        driver.findElement(
                org.openqa.selenium.By.cssSelector("button[type='submit']")
        ).click();
        String pageSource = driver.getPageSource();
        assertTrue(pageSource.contains("required"));
    }

    @Test
    @Order(4)
    void caso4_accesoModuloCitas() {
        driver.get(baseUrl + "/citas");
        CitasPage citasPage = new CitasPage(driver);
        assertTrue(driver.getPageSource().contains("Citas"));
        citasPage.irANuevaCita();
        assertTrue(driver.getCurrentUrl().contains("/citas"));
    }

    @Test
    @Order(5)
    void caso5_navegacionCompleta() {
        driver.get(baseUrl + "/duenios");
        assertTrue(driver.getPageSource().contains("Dueños"));

        driver.get(baseUrl + "/mascotas");
        assertTrue(driver.getPageSource().contains("Mascotas"));

        driver.get(baseUrl + "/citas");
        assertTrue(driver.getPageSource().contains("Citas"));

        driver.get(baseUrl + "/tratamientos");
        assertTrue(driver.getPageSource().contains("Tratamientos"));
    }
}