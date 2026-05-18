package com.panaderia.panaderia.controllers;

import com.panaderia.panaderia.models.Carrito;
import com.panaderia.panaderia.models.Producto;
import com.panaderia.panaderia.repository.IProductoRepository;
import com.panaderia.panaderia.service.ICarritoService;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class CarritoControllerTest {

    @Mock
    private ICarritoService carritoService;

    @Mock
    private IProductoRepository productoRepository;

    @InjectMocks
    private CarritoController carritoController;

    @Test
    void testAddToCartCuandoNoExiste() {
        Producto producto = new Producto();
        producto.setId(1L);
        producto.setPrecio(3000.0);

        Carrito carrito = new Carrito();
        carrito.setProducto(producto);
        carrito.setCantidad(2);

        when(productoRepository.findById(1L)).thenReturn(Optional.of(producto));
        when(carritoService.findByProductoId(1L)).thenReturn(null);
        when(carritoService.save(carrito)).thenReturn(carrito);

        Carrito resultado = carritoController.addToCart(carrito);

        assertNotNull(resultado);
        assertEquals(6000.0, resultado.getTotal());

        verify(productoRepository, times(1)).findById(1L);
        verify(carritoService, times(1)).findByProductoId(1L);
        verify(carritoService, times(1)).save(carrito);
    }

    @Test
    void testAddToCartCuandoYaExiste() {
        Producto producto = new Producto();
        producto.setId(1L);
        producto.setPrecio(3000.0);

        Carrito carritoNuevo = new Carrito();
        carritoNuevo.setProducto(producto);
        carritoNuevo.setCantidad(2);

        Carrito carritoExistente = new Carrito();
        carritoExistente.setProducto(producto);
        carritoExistente.setCantidad(3);

        when(productoRepository.findById(1L)).thenReturn(Optional.of(producto));
        when(carritoService.findByProductoId(1L)).thenReturn(carritoExistente);
        when(carritoService.save(carritoExistente)).thenReturn(carritoExistente);

        Carrito resultado = carritoController.addToCart(carritoNuevo);

        assertNotNull(resultado);
        assertEquals(5, resultado.getCantidad());
        assertEquals(15000.0, resultado.getTotal());

        verify(productoRepository, times(1)).findById(1L);
        verify(carritoService, times(1)).findByProductoId(1L);
        verify(carritoService, times(1)).save(carritoExistente);
    }

    @Test
    void testGetAllItems() {
        Carrito c1 = new Carrito();
        Carrito c2 = new Carrito();

        when(carritoService.findAll()).thenReturn(Arrays.asList(c1, c2));

        List<Carrito> resultado = carritoController.getAllItems();

        assertEquals(2, resultado.size());
        verify(carritoService, times(1)).findAll();
    }

    @Test
    void testUpdateQuantity() {
        Producto producto = new Producto();
        producto.setPrecio(4000.0);

        Carrito carrito = new Carrito();
        carrito.setProducto(producto);
        carrito.setCantidad(1);

        when(carritoService.findById(1L)).thenReturn(Optional.of(carrito));
        when(carritoService.save(carrito)).thenReturn(carrito);

        Carrito resultado = carritoController.updateQuantity(1L, 3);

        assertNotNull(resultado);
        assertEquals(3, resultado.getCantidad());
        assertEquals(12000.0, resultado.getTotal());

        verify(carritoService, times(1)).findById(1L);
        verify(carritoService, times(1)).save(carrito);
    }

    @Test
    void testDeleteFromCart() {
        carritoController.deleteFromCart(1L);

        verify(carritoService, times(1)).delete(1L);
    }

    @Test
    void testVaciarCarrito() {
        carritoController.vaciarCarrito();

        verify(carritoService, times(1)).deleteAll();
    }
}