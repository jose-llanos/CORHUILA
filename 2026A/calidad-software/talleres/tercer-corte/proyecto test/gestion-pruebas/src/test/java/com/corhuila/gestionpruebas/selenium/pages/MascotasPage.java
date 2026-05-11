package com.corhuila.gestionpruebas.selenium.pages;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.FindBy;
import org.openqa.selenium.support.ui.Select;
import java.util.List;

public class MascotasPage extends BasePage {

    @FindBy(css = "a[href='/mascotas/nueva']")
    private WebElement btnNuevaMascota;

    @FindBy(id = "nombre")
    private WebElement inputNombre;

    @FindBy(id = "especie")
    private WebElement selectEspecie;

    @FindBy(id = "raza")
    private WebElement inputRaza;

    @FindBy(id = "edad")
    private WebElement inputEdad;

    @FindBy(id = "peso")
    private WebElement inputPeso;

    @FindBy(id = "duenio.id")
    private WebElement selectDueno;

    @FindBy(css = "button[type='submit']")
    private WebElement btnGuardar;

    @FindBy(css = ".alert-success")
    private WebElement mensajeExito;

    @FindBy(css = "table tbody tr")
    private List<WebElement> filasTabla;

    public MascotasPage(WebDriver driver) {
        super(driver);
    }

    public void irANuevaMascota() {
        driver.get("http://localhost:8081/mascotas");
        btnNuevaMascota.click();
    }

    public void crearMascota(String nombre, String especie, String raza, String edad) {
        inputNombre.sendKeys(nombre);

        Select dropEspecie = new Select(selectEspecie);
        dropEspecie.selectByValue(especie);

        inputRaza.sendKeys(raza);
        inputEdad.sendKeys(edad);
        inputPeso.sendKeys("5");

        Select dropDueno = new Select(selectDueno);
        dropDueno.selectByIndex(1); // primer dueño disponible, sin depender del ID

        btnGuardar.click();
    }

    public boolean existeMensajeExito(String mensaje) {
        return mensajeExito.getText().contains(mensaje);
    }

    public boolean existeMascotaEnTabla(String nombre) {
        for (WebElement fila : filasTabla) {
            if (fila.getText().contains(nombre)) {
                return true;
            }
        }
        return false;
    }
}