package com.pedidos.repository;

import com.pedidos.model.Pedido;

import org.springframework.data.jpa.repository.JpaRepository;

import org.springframework.stereotype.Repository;

import java.util.List;


@Repository

public interface PedidoRepository extends JpaRepository<Pedido, Long> {


    /**

     * Busca todos los pedidos de un producto

     */


    List<Pedido> findByProductoId(Long productoId);


    /**

     * Obtiene pedidos por estado (CREADO, PROCESADO, CANCELADO)

     */


    List<Pedido> findByEstado(String estado);

}
