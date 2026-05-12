package com.pedidos.repository;

import com.pedidos.model.Producto;

import org.springframework.data.jpa.repository.JpaRepository;

import org.springframework.stereotype.Repository;

import java.util.List;


@Repository

public interface ProductoRepository extends JpaRepository<Producto, Long> {


    /**

     * Busca productos por nombre (búsqueda parcial)

     */


    List<Producto> findByNombreContainingIgnoreCase(String nombre);


    /**

     * Obtiene productos con cantidad > 0 (en stock)

     */


    List<Producto> findByCantidadGreaterThan(int cantidad);

}
