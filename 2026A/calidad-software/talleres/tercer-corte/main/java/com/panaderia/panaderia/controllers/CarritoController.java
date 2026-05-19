package com.panaderia.panaderia.controllers;



import com.panaderia.panaderia.models.Carrito;
import com.panaderia.panaderia.models.Producto;
import com.panaderia.panaderia.repository.IProductoRepository;
import com.panaderia.panaderia.service.ICarritoService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@CrossOrigin(origins = "http://localhost:4200")
@RestController
@RequestMapping("/api/carrito")
public class CarritoController {

    @Autowired
    private ICarritoService carritoService;

    @Autowired
    private IProductoRepository productoRepository;

    @PostMapping
    public Carrito addToCart(@RequestBody Carrito carrito) {
        Producto producto = productoRepository.findById(carrito.getProducto().getId())
                .orElseThrow(() -> new RuntimeException("Producto no encontrado"));

        Carrito carritoExistente = carritoService.findByProductoId(producto.getId());

        if (carritoExistente != null) {
            carritoExistente.setCantidad(carritoExistente.getCantidad() + carrito.getCantidad());
            carritoExistente.setTotal(producto.getPrecio() * carritoExistente.getCantidad());
            return carritoService.save(carritoExistente);
        } else {
            carrito.setProducto(producto);
            carrito.setTotal(producto.getPrecio() * carrito.getCantidad());
            return carritoService.save(carrito);
        }
    }

    @GetMapping
    public List<Carrito> getAllItems() {
        return carritoService.findAll();
    }

    @PutMapping("/{id}")
    public Carrito updateQuantity(@PathVariable Long id, @RequestParam int cantidad) {
        Carrito carrito = carritoService.findById(id)
                .orElseThrow(() -> new RuntimeException("Producto en carrito no encontrado"));

        carrito.setCantidad(cantidad);  // Actualiza la cantidad
        carrito.setTotal(carrito.getProducto().getPrecio() * cantidad);  // Recalcula el total

        return carritoService.save(carrito);  // Guarda la actualización
    }

    @DeleteMapping("/{id}")
    public void deleteFromCart(@PathVariable Long id) {
        carritoService.delete(id);
    }

    @DeleteMapping
    public void vaciarCarrito() {
        carritoService.deleteAll();
    }
}
