package com.pedidos.model;

import jakarta.persistence.*;

import java.time.LocalDateTime;


@Entity

@Table(name = "pedidos")

public class Pedido {

    @Id

    @GeneratedValue(strategy = GenerationType.IDENTITY)

    private Long id;

    private Long productoId;

    private int cantidad;

    private double total;

    private String estado;

    private LocalDateTime fecha;


    // Constructores

    public Pedido() {}

    public Pedido(Long productoId, int cantidad, double total, String estado, LocalDateTime fecha) {

        this.productoId = productoId;

        this.cantidad = cantidad;

        this.total = total;

        this.estado = estado;

        this.fecha = fecha;

    }

    // Getters

    public Long getId() { return id; }

    public Long getProductoId() { return productoId; }

    public int getCantidad() { return cantidad; }

    public double getTotal() { return total; }

    public String getEstado() { return estado; }

    public LocalDateTime getFecha() { return fecha; }

    // Setters

    public void setId(Long id) { this.id = id; }

    public void setProductoId(Long productoId) { this.productoId = productoId; }

    public void setCantidad(int cantidad) { this.cantidad = cantidad; }

    public void setTotal(double total) { this.total = total; }

    public void setEstado(String estado) { this.estado = estado; }

    public void setFecha(LocalDateTime fecha) { this.fecha = fecha; }

}
