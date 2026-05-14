package com.sgplab.backend.service;

import com.sgplab.backend.Iservice.IUsuarioService;
import com.sgplab.backend.model.entity.Usuario;
import com.sgplab.backend.repository.IUsuarioRepository;
import org.springframework.stereotype.Service;
import com.sgplab.backend.model.enums.Rol;

import java.util.List;

@Service
public class UsuarioService implements IUsuarioService {

    private final IUsuarioRepository usuarioRepository;

    public UsuarioService(IUsuarioRepository usuarioRepository) {
        this.usuarioRepository = usuarioRepository;
    }

    @Override
    public Usuario obtenerUsuarioPorId(Long id) {
        return usuarioRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Usuario no encontrado con ID: " + id));
    }

    @Override
    public List<Usuario> obtenerTodosLosUsuarios() {
        return usuarioRepository.findAll();
    }

    @Override
    public Usuario actualizarUsuario(Long id, Usuario usuarioDetalles) {
        Usuario usuarioExistente = obtenerUsuarioPorId(id);
        usuarioExistente.setNombre(usuarioDetalles.getNombre());
        usuarioExistente.setEmail(usuarioDetalles.getEmail());

        return usuarioRepository.save(usuarioExistente);
    }

    @Override
    public void eliminarUsuario(Long id) {
        if (!usuarioRepository.existsById(id)) {
            throw new RuntimeException("El usuario con ID " + id + " no existe.");
        }
        usuarioRepository.deleteById(id);
    }
    @Override
    public Usuario crearUsuario(Usuario usuario) {
        if (usuario.getEmail() == null || !usuario.getEmail().contains("@")) {
            throw new IllegalArgumentException("El email proporcionado no es válido.");
        }
        usuario.setRol(Rol.CLIENTE);

        return usuarioRepository.save(usuario);
    }

}