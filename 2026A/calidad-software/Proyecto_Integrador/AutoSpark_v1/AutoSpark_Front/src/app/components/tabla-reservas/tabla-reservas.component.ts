import { Component, OnInit } from '@angular/core';
import { ReservaUsuario } from './ReservaUsuario';
import { TablaServiceService } from './tabla-service.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-tabla-reservas',
  imports: [CommonModule],
  templateUrl: './tabla-reservas.component.html',
  styleUrl: './tabla-reservas.component.css'
})
export class TablaReservasComponent implements OnInit{
  reservas: ReservaUsuario[] = [];
  userRole: string | null = null;

  constructor(private reservaService: TablaServiceService) {}

  ngOnInit(): void {
    this.userRole = localStorage.getItem('userRole');
    this.cargarReservas();
  }

  cargarReservas(): void {
    this.reservaService.getReservasConUsuario().subscribe({
      next: (data) => {
        this.reservas = data;
      },
      error: (err) => {
        console.error('Error al obtener reservas:', err);
      }
    });
  }

}
