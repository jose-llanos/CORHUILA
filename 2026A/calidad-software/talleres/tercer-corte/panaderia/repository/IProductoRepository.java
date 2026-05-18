package com.panaderia.panaderia.repository;



import com.panaderia.panaderia.models.Producto;
import org.springframework.data.repository.CrudRepository;

public interface IProductoRepository extends CrudRepository<Producto, Long> {
}
