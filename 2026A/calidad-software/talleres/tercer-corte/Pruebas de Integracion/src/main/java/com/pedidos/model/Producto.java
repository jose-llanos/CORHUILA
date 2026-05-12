package com.pedidos.model;

import jakarta.persistence.*;


@Entity

@Table(name = "productos")

public class Producto {

    @Id

    @GeneratedValue(strategy = GenerationType.IDENTITY)

    private Long id;

    private String nombre;

    private double precio;

    private int cantidad;



    // Constructores, getters y setters

    public Producto() {}


    public Producto(String nombre, double precio, int cantidad) {

        this.nombre = nombre;

        this.precio = precio;

        this.cantidad = cantidad;

    }


    public Long getId() { return id; }

    public String getNombre() { return nombre; }

    public double getPrecio() { return precio; }

    public int getCantidad() { return cantidad; }

    public void setCantidad(int cantidad) { this.cantidad = cantidad; }

}
