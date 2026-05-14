package com.sgplab.backend.Iservice;

import com.sgplab.backend.model.entity.Usuario;
import java.util.List;

public interface IUsuarioService {
    Usuario obtenerUsuarioPorId(Long id);
    List<Usuario> obtenerTodosLosUsuarios();
    Usuario actualizarUsuario(Long id, Usuario usuarioDetalles);
    void eliminarUsuario(Long id);
    Usuario crearUsuario(Usuario usuario);
}


