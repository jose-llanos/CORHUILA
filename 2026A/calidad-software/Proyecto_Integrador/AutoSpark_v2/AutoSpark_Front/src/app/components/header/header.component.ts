import { CommonModule, isPlatformBrowser } from '@angular/common';
import { Component, Inject, OnInit, PLATFORM_ID } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { NavigationEnd, Router, RouterModule } from '@angular/router';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './header.component.html',
  styleUrl: './header.component.css'
})
export class HeaderComponent implements OnInit {
  userRole: string | null = null;
  isBrowser: boolean;

  constructor(
    private router: Router,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {
    this.isBrowser = isPlatformBrowser(this.platformId);
  }

  ngOnInit(): void {
    if (this.isBrowser) {
      this.cargarRolUsuario();

      this.router.events.subscribe(event => {
        if (event instanceof NavigationEnd) {
          this.cargarRolUsuario();
        }
      });
    }
  }

  cargarRolUsuario(): void {
    this.userRole = localStorage.getItem('userRole');
    console.log('Rol cargado en header:', this.userRole);
  }

  cerrarSesion(): void {
    if (this.isBrowser) {
      localStorage.removeItem('userRole');
      this.userRole = null;
      this.router.navigate(['/login']);
      console.log('Sesión cerrada');
    }
  }

  reservarServicios(): void {
    this.router.navigate(['/reserves']);
  }

  tablaReservas(): void {
    this.router.navigate(['/tablareservas']);
  }

  gestionUsuarios(): void {
    this.router.navigate(['/users-table']);
  }
}