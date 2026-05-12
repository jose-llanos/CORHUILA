package com.pedidos.integration;

import com.pedidos.model.*;

import com.pedidos.repository.*;

import com.pedidos.service.*;

import org.junit.jupiter.api.Test;

import org.springframework.beans.factory.annotation.Autowired;

import org.springframework.boot.test.context.SpringBootTest;

import org.springframework.test.annotation.Rollback;

import org.springframework.transaction.annotation.Transactional;


import static org.junit.jupiter.api.Assertions.*;


@SpringBootTest

@Transactional

public class PedidoIntegrationTest {


    @Autowired

    private PedidoService pedidoService;


    @Autowired

    private ProductoRepository productoRepository;


    @Autowired

    private PedidoRepository pedidoRepository;


    @Test

    void debeCrearPedidoYActualizarInventario() {

        // ARRANGE: Preparar datos de prueba

        Producto producto = new Producto("Laptop", 1200.00, 10);

        Producto guardado = productoRepository.save(producto);


        // ACT: Ejecutar la acción - Crear un pedido

        Pedido pedido = pedidoService.crearPedido(guardado.getId(), 3);


        // ASSERT: Verificar resultados

        assertNotNull(pedido.getId(), "Pedido debe tener ID");

        assertEquals("CREADO", pedido.getEstado(), "Estado debe ser CREADO");

        assertEquals(3600.00, pedido.getTotal(), "Total debe ser 3600");


        // ASSERT: Verificar que el inventario se actualizó

        Producto actualizado = productoRepository.findById(guardado.getId()).get();

        assertEquals(7, actualizado.getCantidad(), "Stock debe reducirse a 7");

    }


    @Test

    void debeRechazarPedidoSinStock() {

        // ARRANGE

        Producto producto = new Producto("Mouse", 25.00, 2);

        Producto guardado = productoRepository.save(producto);


        // ACT & ASSERT: Debe lanzar excepción

        assertThrows(IllegalArgumentException.class, () -> {

            pedidoService.crearPedido(guardado.getId(), 5);

        }, "Debe rechazar pedido sin stock");

    }

}
