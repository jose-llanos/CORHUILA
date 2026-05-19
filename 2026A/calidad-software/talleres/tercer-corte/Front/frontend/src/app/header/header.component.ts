import { Component } from '@angular/core';
import { RouterModule } from '@angular/router';
import { Router } from '@angular/router';
@Component({
  selector: 'app-header',
  standalone: true,
  imports: [RouterModule],
  templateUrl: './header.component.html',
  styleUrl: './header.component.css'
})
export class HeaderComponent {

  isLoggedIn: boolean = false;

  constructor(private router: Router) {}

  ngOnInit(): void {
    // Verificar si el usuario está logueado
    const userToken = localStorage.getItem('authToken'); // Cambia 'authToken' por la clave que uses
    this.isLoggedIn = !!userToken; // Si existe el token, el usuario está logueado
  }


  
  cerrarSesion(): void {
    // Eliminar datos del usuario de localStorage
    localStorage.removeItem('authToken'); // Cambia 'authToken' por la clave que uses
    localStorage.removeItem('userId'); // Opcional: elimina otros datos relacionados
    this.isLoggedIn = false;
    this.router.navigate(['/login']); // Redirigir al login
  }
}
