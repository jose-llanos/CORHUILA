package com.panaderia.panaderia.service;


import com.panaderia.panaderia.models.Carrito;
import java.util.List;
import java.util.Optional;




public interface ICarritoService {
    Carrito save(Carrito carrito);
    void delete(Long id);
    List<Carrito> findAll();
    Optional<Carrito> findById(Long id);
    Carrito findByProductoId(Long productoId);
    void deleteAll();
}
