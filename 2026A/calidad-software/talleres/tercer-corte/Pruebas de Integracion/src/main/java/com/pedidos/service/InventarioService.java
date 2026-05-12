package com.pedidos.service;

import com.pedidos.model.Producto;

import com.pedidos.repository.ProductoRepository;

import org.springframework.stereotype.Service;


@Service

public class InventarioService {


    private final ProductoRepository productoRepository;


    public InventarioService(ProductoRepository productoRepository) {

        this.productoRepository = productoRepository;

    }


    public boolean hayDisponibilidad(Long productoId, int cantidad) {

        Producto producto = productoRepository.findById(productoId).orElse(null);

        return producto != null && producto.getCantidad() >= cantidad;

    }

    public double obtenerPrecio(Long productoId) {

        return productoRepository.findById(productoId)

                .map(Producto::getPrecio)

                .orElseThrow(() -> new IllegalArgumentException("Producto no encontrado"));

    }


    public void actualizarStock(Long productoId, int cantidadVendida) {

        Producto producto = productoRepository.findById(productoId)

                .orElseThrow(() -> new IllegalArgumentException("Producto no encontrado"));

        producto.setCantidad(producto.getCantidad() - cantidadVendida);

        productoRepository.save(producto);

    }

}