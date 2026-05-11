package com.corhuila.gestionpruebas.selenium.pages;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.FindBy;
import org.openqa.selenium.support.ui.Select;
import java.util.List;

public class CitasPage extends BasePage {

    @FindBy(css = "a[href='/citas/nueva']")
    private WebElement btnNuevaCita;

    @FindBy(id = "mascota.id")
    private WebElement selectMascota;

    @FindBy(id = "fecha")
    private WebElement inputFecha;

    @FindBy(id = "motivo")
    private WebElement inputMotivo;

    @FindBy(id = "estado")
    private WebElement selectEstado;

    @FindBy(css = "button[type='submit']")
    private WebElement btnGuardar;

    @FindBy(css = ".alert-success")
    private WebElement mensajeExito;

    @FindBy(css = "table tbody tr")
    private List<WebElement> filasTabla;

    public CitasPage(WebDriver driver) {
        super(driver);
    }

    public void irANuevaCita() {
        driver.get("http://localhost:8081/citas");
        btnNuevaCita.click();
    }

    public void crearCita(String fecha, String hora, String motivo, String mascotaId) {
        Select dropMascota = new Select(selectMascota);
        dropMascota.selectByValue(mascotaId);

        // datetime-local espera formato: "2026-05-10T10:00"
        inputFecha.sendKeys(fecha + "T" + hora);

        inputMotivo.sendKeys(motivo);
        btnGuardar.click();
    }

    public boolean existeMensajeExito(String mensaje) {
        return mensajeExito.getText().contains(mensaje);
    }

    public boolean existeCitaEnTabla(String motivo) {
        for (WebElement fila : filasTabla) {
            if (fila.getText().contains(motivo)) {
                return true;
            }
        }
        return false;
    }
}