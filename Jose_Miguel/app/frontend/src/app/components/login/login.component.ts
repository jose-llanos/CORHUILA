import { Component } from '@angular/core';
import { Router, RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LoginService } from './login.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {

  usuario = {
    email: '',
    password: ''
  };

  mensajeModal: string | null = null;
  mostrarContrasenia: boolean = false;

  constructor(
    private loginService: LoginService,
    private router: Router
  ) {}

  iniciarSesion(): void {
    this.loginService.login(this.usuario).subscribe({

      next: (response: any) => {
        console.log('Inicio de sesión exitoso', response);

        if (response) {
          localStorage.setItem('userRole', response.role);

          if (response.id !== undefined && response.id !== null) {
            localStorage.setItem('userId', response.id.toString());
          }

          this.mensajeModal = '✅ Inicio de sesión exitoso';

          this.router.navigate(['/services']);
        }
      },

      error: (error: any) => {
        console.error('Error en el inicio de sesión', error);

        localStorage.removeItem('userRole');
        localStorage.removeItem('userId');

        this.mensajeModal = '❌ Credenciales incorrectas. Inténtalo de nuevo.';

        this.router.navigate(['/login']);
      }

    });
  }

  toggleMostrarContrasenia(): void {
    this.mostrarContrasenia = !this.mostrarContrasenia;
  }

  cerrarModal(): void {
    this.mensajeModal = null;
  }
}