import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';

import { UsersTable } from './UsersTable';
import { UserServiceService } from './user-service.service';

@Component({
  selector: 'app-users-table',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './users-table.component.html',
  styleUrl: './users-table.component.css'
})
export class UsersTableComponent implements OnInit {

  usuarios: UsersTable[] = [];
  mensajeModal: string = '';

  constructor(private userService: UserServiceService) {}

  ngOnInit(): void {
    this.cargarUsuarios();
  }

cargarUsuarios(): void {

  const userIdLogueado = localStorage.getItem('userId');

  this.userService.getUsers().subscribe({

    next: (data: UsersTable[]) => {

      this.usuarios = data.filter(usuario =>
        usuario.id.toString() !== userIdLogueado
      );

    },

    error: (err: unknown) => {

      console.error('Error cargando usuarios', err);

      this.mensajeModal = 'Error cargando usuarios';
    }

  });
}

  cambiarRol(usuario: UsersTable): void {
    const nuevoRol: 'CUSTOMER' | 'ADMIN' =
      usuario.role === 'CUSTOMER' ? 'ADMIN' : 'CUSTOMER';

    this.userService.changeRole(usuario.id, nuevoRol).subscribe({
      next: () => {
        usuario.role = nuevoRol;

        this.mensajeModal =
          nuevoRol === 'ADMIN'
            ? 'Usuario ascendido a administrador'
            : 'Administrador cambiado a cliente';
      },
      error: (err: unknown) => {
        console.error('Error actualizando rol', err);
        this.mensajeModal = 'No se pudo actualizar el rol';
      }
    });
  }

  cerrarModal(): void {
    this.mensajeModal = '';
  }
}