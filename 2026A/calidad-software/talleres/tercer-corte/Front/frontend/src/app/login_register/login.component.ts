import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterOutlet } from '@angular/router';
import Swal from 'sweetalert2';
import { RouterModule, ActivatedRoute } from '@angular/router'; 
import { LoginService } from './login.service';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {
  email: string = '';
  password: string = '';
  errorMessage: string = '';
  isLoading: boolean = false;
  showPassword: boolean = false;

  private loginService = inject(LoginService);
  private router = inject(Router);
  private http = inject(HttpClient); // Inyecta HttpClient correctamente

  async login(): Promise<void> {
    this.errorMessage = '';

    // Validar que los campos no estén vacíos
    if (!this.email || !this.password) {
      this.errorMessage = 'Por favor, ingrese correo y contraseña';
      return;
    }

    this.isLoading = true;

    try {
      // Validar usuario con el servicio
      const user: { rol: string; id?: string } = await this.loginService.validateUser(this.email, this.password);

    
      

      if (user) {
        // Guardar el rol en el almacenamiento local
        localStorage.setItem('userRole', user.rol);
        if (user.id) {
          localStorage.setItem('userId', user.id); // Guardar el ID del usuario
        } else {
          console.warn('El usuario no tiene un ID definido.');
        }
    // Si tienes un token de autenticación

        // Redirigir a la vista de cliente
        if (user.rol === 'Cliente Nuevo') {
          this.router.navigate(['/clienteview']); // Cambia esta ruta según tu configuración
        } else if (user.rol === 'Administrador') {
          this.router.navigate(['/admin']); // Cambia esta ruta según tu configuración
        } else {
          this.errorMessage = 'Rol no reconocido. Contacte al administrador.';
        }
      } else {
        this.errorMessage = 'Usuario o contraseña incorrectos';
      }
    } catch (error) {
      console.error('Error al intentar iniciar sesión:', error);
      this.errorMessage = 'Error al conectar con el servidor. Intente nuevamente más tarde.';
    } finally {
      this.isLoading = false;
    }
  }

  togglePasswordVisibility(): void {
    this.showPassword = !this.showPassword; // Alterna entre mostrar y ocultar la contraseña
  }
}