import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { RecoverpasswordService } from './recoverpasword.service';

@Component({
  selector: 'app-recoverpassword',
  standalone: true,
  imports: [FormsModule,CommonModule],
  templateUrl: './recoverpassword.component.html',
  styleUrl: './recoverpassword.component.css'
})
export class RecoverpasswordComponent {
  email: string = '';
  mensaje: string = '';
  error: boolean = false;
  mostrarModal: boolean = false;

  constructor(private recoverpaswordservice: RecoverpasswordService, private router: Router) {}

  recuperarContrasenia() {
    if (!this.email.trim()) {
      this.mensaje = 'Por favor, ingresa un correo válido.';
      this.error = true;
      return;
    }

    this.recoverpaswordservice.recuperarContrasenia(this.email).subscribe({
      next: (respuesta) => {
        this.mensaje = 'Revisa tu correo para continuar con la recuperación.';
        this.error = false;
        this.mostrarModal = true;
      },
      error: (err) => {
        this.mensaje = err.error || 'Ocurrió un error al recuperar la contraseña.';
        this.error = true;
        this.mostrarModal = true;
      }
    });
  }

  cerrarModal() {
    this.mostrarModal = false;
    this.router.navigate(['/login']);
  }
}
