import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { RecuperarContraseniaService } from './recuperarcontrasenia.service';

@Component({
  selector: 'app-recuperarcontrasenia',
  standalone: true,
  imports: [FormsModule,CommonModule],
  templateUrl: './recuperarcontrasenia.component.html',
  styleUrls: ['./recuperarcontrasenia.component.css']
})
export class RecuperarcontraseniaComponent {
  email: string = '';
  mensaje: string = '';
  error: boolean = false;
  mostrarModal: boolean = false;

  constructor(private recuperarContraseniaService: RecuperarContraseniaService, private router: Router) {}

  recuperarContrasenia() {
    if (!this.email.trim()) {
      this.mensaje = 'Por favor, ingresa un correo válido.';
      this.error = true;
      return;
    }

    this.recuperarContraseniaService.recuperarContrasenia(this.email).subscribe({
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
