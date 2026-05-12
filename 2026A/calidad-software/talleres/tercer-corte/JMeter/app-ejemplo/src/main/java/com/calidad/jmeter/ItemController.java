package com.calidad.jmeter;

import org.springframework.web.bind.annotation.*;
import java.util.*;

/**
 * Controlador REST con múltiples endpoints para pruebas de carga
 */
@RestController
@RequestMapping("/api/v1")
public class ItemController {

    // Simulamos una pequeña base de datos en memoria
    private static final List<Item> items = new ArrayList<>();

    static {
        items.add(new Item(1L, "Laptop", 999.99));
        items.add(new Item(2L, "Mouse", 25.50));
        items.add(new Item(3L, "Teclado", 75.00));
        items.add(new Item(4L, "Monitor", 299.99));
        items.add(new Item(5L, "Webcam", 59.99));
    }

    /**
     * Endpoint GET simple - lista todos los items
     * Método: GET
     * Ruta: http://localhost:8080/api/v1/items
     */
    @GetMapping("/items")
    public List<Item> obtenerItems() {
        return items;
    }

    /**
     * Endpoint GET con parámetro - obtiene un item específico
     * Método: GET
     * Ruta: http://localhost:8080/api/v1/items/{id}
     */
    @GetMapping("/items/{id}")
    public Item obtenerItem(@PathVariable Long id) {
        return items.stream()
                .filter(item -> item.getId().equals(id))
                .findFirst()
                .orElse(new Item(-1L, "No encontrado", 0.0));
    }

    /**
     * Endpoint POST - crear un nuevo item
     * Método: POST
     * Ruta: http://localhost:8080/api/v1/items
     * Body: {"nombre": "Producto", "precio": 99.99}
     */
    @PostMapping("/items")
    public Item crearItem(@RequestBody Item nuevoItem) {
        Long nuevoId = items.stream()
                .mapToLong(Item::getId)
                .max()
                .orElse(0) + 1;
        nuevoItem.setId(nuevoId);
        items.add(nuevoItem);
        return nuevoItem;
    }

    /**
     * Endpoint GET con simulación de carga - tarda 2 segundos
     * Método: GET
     * Ruta: http://localhost:8080/api/v1/items/report
     */
    @GetMapping("/items/report")
    public Map<String, Object> generarReporte() throws InterruptedException {
        // Simulamos procesamiento lento
        Thread.sleep(2000);
        
        Map<String, Object> reporte = new LinkedHashMap<>();
        reporte.put("total", items.size());
        reporte.put("items", items);
        reporte.put("timestamp", System.currentTimeMillis());
        return reporte;
    }

    /**
     * Endpoint GET para verificar salud de la aplicación
     * Método: GET
     * Ruta: http://localhost:8080/api/v1/health
     */
    @GetMapping("/health")
    public Map<String, String> verificarSalud() {
        Map<String, String> health = new LinkedHashMap<>();
        health.put("status", "UP");
        health.put("version", "1.0.0");
        health.put("message", "Aplicación funcionando correctamente");
        return health;
    }

    /**
     * Endpoint PUT - actualizar un item
     * Método: PUT
     * Ruta: http://localhost:8080/api/v1/items/{id}
     * Body: {"nombre": "Nuevo nombre", "precio": 149.99}
     */
    @PutMapping("/items/{id}")
    public Item actualizarItem(@PathVariable Long id, @RequestBody Item itemActualizado) {
        return items.stream()
                .filter(item -> item.getId().equals(id))
                .findFirst()
                .map(item -> {
                    item.setNombre(itemActualizado.getNombre());
                    item.setPrecio(itemActualizado.getPrecio());
                    return item;
                })
                .orElse(null);
    }

    /**
     * Endpoint DELETE - eliminar un item
     * Método: DELETE
     * Ruta: http://localhost:8080/api/v1/items/{id}
     */
    @DeleteMapping("/items/{id}")
    public Map<String, String> eliminarItem(@PathVariable Long id) {
        items.removeIf(item -> item.getId().equals(id));
        Map<String, String> respuesta = new LinkedHashMap<>();
        respuesta.put("mensaje", "Item eliminado");
        respuesta.put("id", id.toString());
        return respuesta;
    }

    /**
     * Clase modelo para Item
     */
    public static class Item {
        private Long id;
        private String nombre;
        private Double precio;

        public Item() {}

        public Item(Long id, String nombre, Double precio) {
            this.id = id;
            this.nombre = nombre;
            this.precio = precio;
        }

        // Getters y Setters
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }

        public String getNombre() { return nombre; }
        public void setNombre(String nombre) { this.nombre = nombre; }

        public Double getPrecio() { return precio; }
        public void setPrecio(Double precio) { this.precio = precio; }
    }
}
