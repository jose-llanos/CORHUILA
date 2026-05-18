package com.panaderia.panaderia.service;

import com.panaderia.panaderia.models.Carrito;
import com.panaderia.panaderia.models.Producto;
import com.panaderia.panaderia.repository.CarritoRepository;

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
class CarritoServiceImplTest {

    @Mock
    private CarritoRepository carritoRepository;

    @InjectMocks
    private CarritoServiceImpl carritoService;

    @Test
    void testSave() {

        Producto producto = new Producto();
        producto.setPrecio(5000.0);

        Carrito carrito = new Carrito();
        carrito.setProducto(producto);
        carrito.setCantidad(2);

        when(carritoRepository.save(any(Carrito.class))).thenReturn(carrito);

        Carrito resultado = carritoService.save(carrito);

        assertNotNull(resultado);
        assertEquals(10000.0, resultado.getTotal());

        verify(carritoRepository, times(1)).save(carrito);
    }

    @Test
    void testFindAll() {

        Carrito c1 = new Carrito();
        Carrito c2 = new Carrito();

        when(carritoRepository.findAll()).thenReturn(Arrays.asList(c1, c2));

        List<Carrito> resultado = carritoService.findAll();

        assertEquals(2, resultado.size());

        verify(carritoRepository, times(1)).findAll();
    }

    @Test
    void testFindById() {

        Carrito carrito = new Carrito();

        when(carritoRepository.findById(1L))
                .thenReturn(Optional.of(carrito));

        Optional<Carrito> resultado =
                carritoService.findById(1L);

        assertTrue(resultado.isPresent());

        verify(carritoRepository, times(1))
                .findById(1L);
    }

    @Test
    void testFindByProductoId() {

        Carrito carrito = new Carrito();

        when(carritoRepository.findByProductoId(1L))
                .thenReturn(carrito);

        Carrito resultado =
                carritoService.findByProductoId(1L);

        assertNotNull(resultado);

        verify(carritoRepository, times(1))
                .findByProductoId(1L);
    }

    @Test
    void testDelete() {

        carritoService.delete(1L);

        verify(carritoRepository, times(1))
                .deleteById(1L);
    }

    @Test
    void testDeleteAll() {

        carritoService.deleteAll();

        verify(carritoRepository, times(1))
                .deleteAll();
    }

}