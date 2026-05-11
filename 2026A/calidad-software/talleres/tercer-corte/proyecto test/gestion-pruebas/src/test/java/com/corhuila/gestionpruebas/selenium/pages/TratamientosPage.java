package com.corhuila.gestionpruebas.selenium.pages;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.FindBy;
import org.openqa.selenium.support.ui.Select;
import java.util.List;

public class TratamientosPage extends BasePage {

    @FindBy(css = "a[href*='/tratamientos/new']")
    private WebElement btnNuevoTratamiento;

    @FindBy(id = "descripcion")
    private WebElement inputDescripcion;

    @FindBy(id = "medicamento")
    private WebElement inputMedicamento;

    @FindBy(id = "dosis")
    private WebElement inputDosis;

    @FindBy(id = "duracion")
    private WebElement inputDuracion;

    @FindBy(id = "citaId")
    private WebElement selectCita;

    @FindBy(css = "button[type='submit']")
    private WebElement btnGuardar;

    @FindBy(css = ".alert-success")
    private WebElement mensajeExito;

    @FindBy(css = "table tbody tr")
    private List<WebElement> filasTabla;

    public TratamientosPage(WebDriver driver) {
        super(driver);
    }

    public void irANuevoTratamiento() {
        btnNuevoTratamiento.click();
    }

    public void crearTratamiento(String descripcion, String medicamento, String dosis, String duracion, String citaId) {
        inputDescripcion.sendKeys(descripcion);
        inputMedicamento.sendKeys(medicamento);
        inputDosis.sendKeys(dosis);
        inputDuracion.sendKeys(duracion);

        Select dropdown = new Select(selectCita);
        dropdown.selectByValue(citaId);

        btnGuardar.click();
    }

    public boolean existeMensajeExito(String mensaje) {
        return mensajeExito.getText().contains(mensaje);
    }

    public boolean existeTratamientoEnTabla(String descripcion) {
        for (WebElement fila : filasTabla) {
            if (fila.getText().contains(descripcion)) {
                return true;
            }
        }
        return false;
    }
}