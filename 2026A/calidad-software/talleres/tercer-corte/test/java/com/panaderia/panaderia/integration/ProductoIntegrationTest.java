package com.panaderia.panaderia.integration;

import com.panaderia.panaderia.models.Producto;
import com.panaderia.panaderia.repository.IProductoRepository;
import com.panaderia.panaderia.service.IProductoService;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

import org.springframework.test.context.ActiveProfiles;

@SpringBootTest
@Transactional
@ActiveProfiles("test")
class ProductoIntegrationTest {

    @Autowired
    private IProductoService productoService;

    @Autowired
    private IProductoRepository productoRepository;

    @Test
    void debeGuardarYListarProductos() {

        // ARRANGE
        Producto producto = new Producto();
        producto.setNombre("Pan de queso");
        producto.setDescripcion("Producto de prueba");
        producto.setPrecio(2500.0);
        producto.setCantidad(10);
        producto.setImagenUrl("pan.jpg");

        // ACT
        Producto guardado = productoService.save(producto);
        List<Producto> productos = productoService.findAll();

        // ASSERT
        assertNotNull(guardado.getId());
        assertFalse(productos.isEmpty());
        assertEquals("Pan de queso", guardado.getNombre());
    }

    @Test
    void debeBuscarProductoPorId() {

        // ARRANGE
        Producto producto = new Producto();
        producto.setNombre("Croissant");
        producto.setDescripcion("Producto de prueba");
        producto.setPrecio(4000.0);
        producto.setCantidad(5);
        producto.setImagenUrl("croissant.jpg");

        Producto guardado =
                productoRepository.save(producto);

        // ACT
        Producto encontrado =
                productoService.findById(guardado.getId());

        // ASSERT
        assertNotNull(encontrado);
        assertEquals("Croissant", encontrado.getNombre());
    }

    @Test
    void debeEliminarProducto() {

        // ARRANGE
        Producto producto = new Producto();
        producto.setNombre("Buñuelo");
        producto.setDescripcion("Producto de prueba");
        producto.setPrecio(1500.0);
        producto.setCantidad(20);
        producto.setImagenUrl("bunuelo.jpg");

        Producto guardado =
                productoRepository.save(producto);

        // ACT
        productoService.delete(guardado.getId());

        Producto eliminado =
                productoService.findById(guardado.getId());

        // ASSERT
        assertNull(eliminado);
    }
}