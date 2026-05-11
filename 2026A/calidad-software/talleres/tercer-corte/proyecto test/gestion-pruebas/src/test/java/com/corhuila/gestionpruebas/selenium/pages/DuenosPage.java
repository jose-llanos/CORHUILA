package com.corhuila.gestionpruebas.selenium.pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.FindBy;

import java.util.List;

public class DuenosPage extends BasePage {

    @FindBy(css = "a[href='/duenios/nuevo']")
    private WebElement btnNuevoDueno;

    @FindBy(name = "nombre")
    private WebElement inputNombre;

    @FindBy(name = "correo")
    private WebElement inputCorreo;

    @FindBy(name = "telefono")
    private WebElement inputTelefono;

    @FindBy(css = "button[type='submit']")
    private WebElement btnGuardar;

    @FindBy(css = "table tbody tr")
    private List<WebElement> filasTabla;

    public DuenosPage(WebDriver driver) {
        super(driver);
    }

    public void irANuevoDueno() {
        btnNuevoDueno.click();
    }

    public void crearDueno(String nombre, String telefono, String correo) {

        inputNombre.sendKeys(nombre);

        inputCorreo.sendKeys(correo);

        inputTelefono.sendKeys(telefono);

        btnGuardar.click();
    }

    public boolean existeDuenoEnTabla(String nombre) {

        for (WebElement fila : filasTabla) {

            if (fila.getText().contains(nombre)) {
                return true;
            }
        }

        return false;
    }

    public void editarDueno(String nombreActual, String nuevoNombre) {

        WebElement btnEditar = driver.findElement(
                By.xpath("//td[contains(text(),'" + nombreActual +
                        "')]/following-sibling::td//a[contains(text(),'Editar')]")
        );

        btnEditar.click();

        inputNombre.clear();

        inputNombre.sendKeys(nuevoNombre);

        btnGuardar.click();
    }

    public void eliminarDueno(String nombre) {

        WebElement btnEliminar = driver.findElement(
                By.xpath("//td[contains(text(),'" + nombre +
                        "')]/following-sibling::td//button[contains(text(),'Eliminar')]")
        );

        btnEliminar.click();
    }
}