package com.panaderia.panaderia.repository;



import com.panaderia.panaderia.models.Carrito;
import org.springframework.data.repository.CrudRepository;

public interface CarritoRepository extends CrudRepository<Carrito, Long> {
   Carrito findByProductoId(Long productoId);  // Método para encontrar carrito por producto

}
