import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { Router, RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { Usuario } from './usuario';
import { RegisterService } from './register.service';

@Component({
  selector: 'app-register',
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './register.component.html',
  styleUrl: './register.component.css'
})
export class RegisterComponent implements OnInit {
  mostrarContrasenia: boolean = false;
  mensajeModal: String | null = null;

  nuevoUsuario: Usuario = {
    id: 0,
    fullName: '',
    identityCard: '',
    email: '',
    phone: '',
    password: '',
    licensePlate: '',
    role: 'CUSTOMER'
  };

  constructor(private serviciosUsuario: RegisterService, private router: Router) {}

  ngOnInit(): void {}

  registrarUsuario(): void {
    this.serviciosUsuario.create(this.nuevoUsuario).subscribe(
      response => {
        console.log('Usuario registrado exitosamente', response);
        this.mensajeModal = '¡Registro exitoso! Ahora puedes iniciar sesión.';
      },
      error => {
        console.error('Error al registrar usuario', error);
        this.mensajeModal = 'Hubo un error al registrar el usuario. Intenta nuevamente.';
      }
    );
  }

  toggleMostrarContrasenia(): void {
    this.mostrarContrasenia = !this.mostrarContrasenia;
  }

  cerrarModal(): void {
    this.mensajeModal = null;
    this.router.navigate(['/login']);
  }
}