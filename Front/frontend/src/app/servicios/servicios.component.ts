import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { Router, RouterModule } from '@angular/router';

@Component({
  selector: 'app-servicios',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './servicios.component.html',
  styleUrl: './servicios.component.css'
})
export class ServiciosComponent implements OnInit {
  
  servicios: any[] = [
    {
      id: 1,
      nombre: 'Mensualidad',
      descripcion: 'Accede a tarifas preferenciales con nuestros planes de mensualidad.',
      detalles: 'Con nuestras mensualidades, puedes estacionar tu vehículo sin preocupaciones durante todo el mes.',
      imagen: 'https://d2j6dbq0eux0bg.cloudfront.net/images/84149019/4619134130.png'
    },
    {
      id: 2,
      nombre: 'Servicio Nocturno',
      descripcion: 'Estaciona tu vehículo con seguridad toda la noche.',
      detalles: 'Nuestro servicio nocturno garantiza la seguridad de tu vehículo durante la noche.',
      imagen: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSPUXwsgoImR5pZ1crx5oX6VpQgE_paQYrM5Q&s'
    },
    {
      id: 3,
      nombre: 'Techado',
      descripcion: 'Protege tu auto con nuestras plazas de estacionamiento techadas.',
      detalles: 'Nuestras plazas techadas protegen tu vehículo de las inclemencias del tiempo.',
      imagen: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSkoD_PGDnmvU0-ziWlqM2Pmj6ARaIcSl1HxA&s'
    },
    {
      id: 4,
      nombre: 'Valet Parking',
      descripcion: 'Disfruta de un servicio exclusivo de valet parking.',
      detalles: 'Nuestro valet parking ofrece comodidad y exclusividad para nuestros clientes.',
      imagen: 'https://azulreceptionhall.com/wp-content/uploads/2023/09/parking-valet.jpg'
    }
  ];
  // Propiedad para almacenar el servicio seleccionado

  isAdmin: boolean = false; // Verifica si el usuario es administrador
 
 
 
  constructor(private router: Router) {}


  ngOnInit(): void {
    this.checkAdminRole();
  }

  checkAdminRole(): void {
    const userRole = localStorage.getItem('userRole');
    this.isAdmin = userRole === 'Administrador';
  }

  verDetalles(id: number): void {
    this.router.navigate([`/servicio/${id}`]);
  }
  eliminarServicio(id: number): void {
    this.servicios = this.servicios.filter(servicio => servicio.id !== id);
  }

   // Función para auditar (solo visible para administradores)
   auditarServicio(id: number): void {
    alert(`Auditoría iniciada para el servicio con ID: ${id}`);
  }

}
