package com.tasks.app.e2e.pages;
 
import org.openqa.selenium.NoAlertPresentException;
import org.openqa.selenium.TimeoutException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
 
public class RegisterPage extends BasePage {
 
    public RegisterPage(WebDriver driver, WebDriverWait wait, String baseUrl) {
        super(driver, wait, baseUrl);
    }
 
    /**
     * Llena el formulario y envía.
     *
     * <p>El frontend muestra un {@code alert()} nativo del navegador
     * ("Cuenta creada con éxito. Inicia sesión.") tras el registro. Debemos
     * aceptarlo antes de seguir, o cualquier interacción posterior lanzará
     * {@code UnhandledAlertException}.</p>
     */
    public LoginPage register(String username, String email, String password) {
        type(byTest("reg-username"), username);
        type(byTest("reg-email"), email);
        type(byTest("reg-password"), password);
        click(byTest("btn-register-submit"));
 
        acceptAlertIfPresent();
 
        // Tras el alert, el frontend puede:
        //  (a) redirigir solo al login -> ya veremos login-username
        //  (b) seguir en la pantalla de registro -> hay que pulsar link-to-login
        if (isPresent(byTest("login-username"))) {
            return new LoginPage(driver, wait, baseUrl);
        }
        return goToLogin();
    }
 
    public LoginPage goToLogin() {
        click(byTest("link-to-login"));
        waitVisible(byTest("login-username"));
        return new LoginPage(driver, wait, baseUrl);
    }
 
    /**
     * Si hay un alert abierto, lo acepta. Espera hasta 5 s a que aparezca
     * porque el JS puede tardar un instante en dispararlo después del click.
     */
    private void acceptAlertIfPresent() {
        try {
            wait.until(ExpectedConditions.alertIsPresent()).accept();
        } catch (TimeoutException | NoAlertPresentException ignored) {
            // No hubo alert: el frontend no lo mostró. Seguimos.
        }
    }
}