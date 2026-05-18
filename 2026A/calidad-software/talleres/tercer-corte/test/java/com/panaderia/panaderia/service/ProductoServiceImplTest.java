package com.panaderia.panaderia.service;

import com.panaderia.panaderia.models.Producto;
import com.panaderia.panaderia.repository.IProductoRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ProductoServiceImplTest {

    @Mock
    private IProductoRepository productoRepository;

    @InjectMocks
    private ProductoServiceImpl productoService;

    @Test
    void testFindAll() {
        Producto producto1 = new Producto();
        Producto producto2 = new Producto();

        when(productoRepository.findAll()).thenReturn(Arrays.asList(producto1, producto2));

        List<Producto> resultado = productoService.findAll();

        assertEquals(2, resultado.size());
        verify(productoRepository, times(1)).findAll();
    }

    @Test
    void testFindByIdCuandoExiste() {
        Producto producto = new Producto();

        when(productoRepository.findById(1L)).thenReturn(Optional.of(producto));

        Producto resultado = productoService.findById(1L);

        assertNotNull(resultado);
        verify(productoRepository, times(1)).findById(1L);
    }

    @Test
    void testFindByIdCuandoNoExiste() {
        when(productoRepository.findById(99L)).thenReturn(Optional.empty());

        Producto resultado = productoService.findById(99L);

        assertNull(resultado);
        verify(productoRepository, times(1)).findById(99L);
    }

    @Test
    void testSave() {
        Producto producto = new Producto();

        when(productoRepository.save(producto)).thenReturn(producto);

        Producto resultado = productoService.save(producto);

        assertNotNull(resultado);
        verify(productoRepository, times(1)).save(producto);
    }

    @Test
    void testDelete() {
        Long id = 1L;

        doNothing().when(productoRepository).deleteById(id);

        productoService.delete(id);

        verify(productoRepository, times(1)).deleteById(id);
    }
}