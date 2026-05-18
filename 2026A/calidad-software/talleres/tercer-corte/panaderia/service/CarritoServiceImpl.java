package com.panaderia.panaderia.service;


import com.panaderia.panaderia.models.Carrito;
import com.panaderia.panaderia.repository.CarritoRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class CarritoServiceImpl implements ICarritoService {

    @Autowired
    private CarritoRepository carritoRepository;

    @Override
    public Carrito save(Carrito carrito) {
        double total = carrito.getProducto().getPrecio() * carrito.getCantidad();
        carrito.setTotal(total);
        return carritoRepository.save(carrito);
    }

    @Override
    public void delete(Long id) {
        carritoRepository.deleteById(id);
    }
    @Override
    public List<Carrito> findAll() {
        try {
            return (List<Carrito>) carritoRepository.findAll();
        } catch (Exception e) {
            // Manejo de errores, puedes registrar o lanzar excepciones específicas si es necesario
            throw new RuntimeException("Error al obtener los elementos del carrito", e);
        }
    }

    @Override
    public Optional<Carrito> findById(Long id) {
        return carritoRepository.findById(id);
    }

    @Override
    public Carrito findByProductoId(Long productoId) {
        return carritoRepository.findByProductoId(productoId);
    }

    @Override
    public void deleteAll() {
        carritoRepository.deleteAll();
    }
}
