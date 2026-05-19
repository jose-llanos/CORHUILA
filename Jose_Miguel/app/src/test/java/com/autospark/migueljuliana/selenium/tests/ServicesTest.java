package com.autospark.migueljuliana.selenium.tests;

import com.autospark.migueljuliana.selenium.pages.ServicesPage;
import com.aventstack.extentreports.Status;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ServicesTest extends BaseTest {

    /**
     * TC-FUNC-004: Visualización de servicios disponibles
     */
    @Test(description = "TC-FUNC-004: Visualización de servicios disponibles")
    public void testViewServices() {

        test = extent.createTest("Ver servicios disponibles");

        ServicesPage servicesPage = new ServicesPage(driver);

        servicesPage.navigateTo();

        test.log(Status.INFO, "Página de servicios cargada");

        int serviceCount = servicesPage.getServiceCount();

        test.log(
                Status.INFO,
                "Número de servicios encontrados: " + serviceCount
        );

        Assert.assertTrue(
                serviceCount > 0,
                "Debería haber al menos un servicio visible"
        );

        boolean lavadoBasicoVisible =
                servicesPage.isServiceDisplayed("Lavado Basico");

        Assert.assertTrue(
                lavadoBasicoVisible,
                "El servicio 'Lavado Basico' debería estar visible"
        );

        test.log(
                Status.PASS,
                "Servicios visibles correctamente - "
                        + serviceCount
                        + " servicios encontrados"
        );
    }
}