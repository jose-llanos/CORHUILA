package com.pedidos.service;

import com.pedidos.model.*;

import com.pedidos.repository.*;

import org.springframework.stereotype.Service;

import java.time.LocalDateTime;


@Service

public class PedidoService {


    private final PedidoRepository pedidoRepository;

    private final InventarioService inventarioService;


    public PedidoService(PedidoRepository pedidoRepository,

                         InventarioService inventarioService) {

        this.pedidoRepository = pedidoRepository;

        this.inventarioService = inventarioService;

    }


    public Pedido crearPedido(Long productoId, int cantidad) {

        // 1. Validar disponibilidad en inventario

        if (!inventarioService.hayDisponibilidad(productoId, cantidad)) {

            throw new IllegalArgumentException("Producto sin stock");

        }


        // 2. Obtener precio del producto

        double precioUnitario = inventarioService.obtenerPrecio(productoId);


        // 3. Crear pedido

        Pedido pedido = new Pedido();

        pedido.setProductoId(productoId);

        pedido.setCantidad(cantidad);

        pedido.setTotal(precioUnitario * cantidad);

        pedido.setEstado("CREADO");

        pedido.setFecha(LocalDateTime.now());


        // 4. Guardar en BD y actualizar inventario

        pedidoRepository.save(pedido);

        inventarioService.actualizarStock(productoId, cantidad);


        return pedido;

    }

}

