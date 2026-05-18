package com.panaderia.panaderia.service;





import com.panaderia.panaderia.models.Producto;

import java.util.List;

public interface IProductoService {

    List<Producto> findAll();

    Producto findById(Long id);

    Producto save(Producto producto);

    void delete(Long id);
}
