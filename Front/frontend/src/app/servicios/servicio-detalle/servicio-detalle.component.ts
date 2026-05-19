import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { Router, RouterLink } from '@angular/router';

@Component({
  selector: 'app-servicio-detalle',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './servicio-detalle.component.html',
  styleUrl: './servicio-detalle.component.css'
})
export class ServicioDetalleComponent {

}
