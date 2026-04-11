import com.corhuila.calidad.Convertidor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Clase de pruebas unitarias para la clase {@link Convertidor}.
 * <p>
 * Verifica el correcto funcionamiento de las conversiones de temperatura
 * y la validación de entradas numéricas.
 * </p>
 * @version 1.0
 */
public class ConvertidorTest {

    /**
     * Instancia del convertidor utilizada en las pruebas.
     */
    private Convertidor conv = new Convertidor();

    /**
     * Prueba la conversión de Celsius a Fahrenheit.
     * <p><b>Requerimiento:</b> RF-001</p>
     */
    @Test
    public void testCelsiusAFahrenheit() {
        assertEquals(32.0, conv.celsiusAFahrenheit(0), 0.1);
        assertEquals(77.0, conv.celsiusAFahrenheit(25), 0.1);
        assertEquals(14.0, conv.celsiusAFahrenheit(-10), 0.1);
        assertEquals(98.6, conv.celsiusAFahrenheit(37), 0.1);
        assertEquals(97.7, conv.celsiusAFahrenheit(36.5), 0.1);
    }

    /**
     * Prueba la conversión de Fahrenheit a Celsius.
     * <p><b>Requerimiento:</b> RF-002</p>
     */
    @Test
    public void testFahrenheitACelsius() {
        assertEquals(0.0, conv.fahrenheitACelsius(32), 0.1);
        assertEquals(20.0, conv.fahrenheitACelsius(68), 0.1);
        assertEquals(-20.0, conv.fahrenheitACelsius(-4), 0.1);
    }

    /**
     * Prueba la validación de entradas numéricas.
     * <p><b>Requerimiento:</b> RF-003</p>
     */
    @Test
    public void testEsNumerico() {
        assertTrue(conv.esNumerico("25"));
        assertTrue(conv.esNumerico("-10"));
        assertFalse(conv.esNumerico("abc"));
        assertFalse(conv.esNumerico("      "));
    }
}