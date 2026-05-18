package com.panaderia.panaderia.controllers;

import com.panaderia.panaderia.models.Producto;
import com.panaderia.panaderia.service.IProductoService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ProductoControllerTest {

    @Mock
    private IProductoService productoService;

    @InjectMocks
    private ProductoController productoController;

    @Test
    void testIndex() {
        Producto p1 = new Producto();
        Producto p2 = new Producto();

        when(productoService.findAll()).thenReturn(Arrays.asList(p1, p2));

        List<Producto> resultado = productoController.index();

        assertEquals(2, resultado.size());
        verify(productoService, times(1)).findAll();
    }

    @Test
    void testShow() {
        Producto producto = new Producto();
        producto.setId(1L);

        when(productoService.findById(1L)).thenReturn(producto);

        Producto resultado = productoController.show(1L);

        assertNotNull(resultado);
        assertEquals(1L, resultado.getId());
        verify(productoService, times(1)).findById(1L);
    }

    @Test
    void testCreate() {
        Producto producto = new Producto();
        producto.setNombre("Pan");

        when(productoService.save(producto)).thenReturn(producto);

        Producto resultado = productoController.create(producto);

        assertNotNull(resultado);
        assertEquals("Pan", resultado.getNombre());
        verify(productoService, times(1)).save(producto);
    }

    @Test
    void testUpdate() {
        Producto productoActual = new Producto();
        productoActual.setId(1L);
        productoActual.setNombre("Pan viejo");
        productoActual.setDescripcion("Descripción vieja");
        productoActual.setPrecio(1000.0);
        productoActual.setCantidad(5);
        productoActual.setImagenUrl("vieja.jpg");

        Producto productoNuevo = new Producto();
        productoNuevo.setNombre("Pan nuevo");
        productoNuevo.setDescripcion("Descripción nueva");
        productoNuevo.setPrecio(2000.0);
        productoNuevo.setCantidad(10);
        productoNuevo.setImagenUrl("nueva.jpg");

        when(productoService.findById(1L)).thenReturn(productoActual);
        when(productoService.save(productoActual)).thenReturn(productoActual);

        Producto resultado = productoController.update(productoNuevo, 1L);

        assertEquals("Pan nuevo", resultado.getNombre());
        assertEquals("Descripción nueva", resultado.getDescripcion());
        assertEquals(2000.0, resultado.getPrecio());
        assertEquals(10, resultado.getCantidad());
        assertEquals("nueva.jpg", resultado.getImagenUrl());

        verify(productoService, times(1)).findById(1L);
        verify(productoService, times(1)).save(productoActual);
    }

    @Test
    void testDelete() {
        productoController.delete(1L);

        verify(productoService, times(1)).delete(1L);
    }
}