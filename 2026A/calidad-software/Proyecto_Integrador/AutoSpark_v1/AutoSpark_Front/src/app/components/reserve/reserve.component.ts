import { Component, OnInit } from '@angular/core';
import { reserva } from './Reserva';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
import { ReservesServiceService } from './reserves-service.service';
import { Services } from '../services/Service';

@Component({
  selector: 'app-reserve',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './reserve.component.html',
  styleUrl: './reserve.component.css'
})
export class ReserveComponent implements OnInit {
  reservaForm: reserva = {
    id: 0,
    vehicleType: 'CARRO',
    serviceType: '',
    licensePlate: '',
    value: 0,
    reservationDate: '',
    active: true
  };

  mensajeModal: string = '';
  tiposServicio: Services[] = [];
  servicioSeleccionado: Services | null = null;

  fechasOcupadas: string[] = [];
  fechaOcupada: boolean = false;
  fechaMinima: string = '';

  constructor(private reservasService: ReservesServiceService) {}

  ngOnInit(): void {
    this.fechaMinima = this.obtenerFechaMinima();
    this.cargarTiposServicio();
    this.cargarFechasOcupadas();
  }

  cargarTiposServicio(): void {
    this.reservasService.getTiposServicio().subscribe({
      next: (data) => {
        this.tiposServicio = data;
      },
      error: (err) => {
        console.error('Error al obtener tipos de servicio', err);
      }
    });
  }

  cargarFechasOcupadas(): void {
    this.reservasService.getFechasOcupadas().subscribe({
      next: (fechas) => {
        this.fechasOcupadas = fechas.map(fecha => fecha.slice(0, 16));
      },
      error: (err) => {
        console.error('Error al obtener fechas ocupadas', err);
      }
    });
  }

  obtenerFechaMinima(): string {
    const ahora = new Date();
    ahora.setMinutes(ahora.getMinutes() - ahora.getTimezoneOffset());
    return ahora.toISOString().slice(0, 16);
  }

  actualizarPrecio(): void {
    if (this.servicioSeleccionado) {
      this.reservaForm.serviceType = this.servicioSeleccionado.name;
      this.reservaForm.value = this.servicioSeleccionado.price;
    }
  }

  validarFechaSeleccionada(): void {
    const fechaSeleccionada = this.reservaForm.reservationDate;

    this.fechaOcupada = this.fechasOcupadas.includes(fechaSeleccionada);

    if (this.fechaOcupada) {
      this.mensajeModal = 'Ese horario ya está ocupado. Por favor selecciona otra fecha u hora.';
    }
  }

  registrarReserva(): void {
    this.validarFechaSeleccionada();

    if (this.fechaOcupada) {
      this.mensajeModal = 'No puedes registrar una reserva en un horario ocupado.';
      return;
    }

    this.reservaForm.id = 0;
    this.reservaForm.active = true;

    this.reservasService.create(this.reservaForm).subscribe({
      next: () => {
        this.mensajeModal = 'Reserva registrada exitosamente';

        this.reservaForm = {
          id: 0,
          vehicleType: 'CARRO',
          serviceType: '',
          value: 0,
          licensePlate: '',
          reservationDate: '',
          active: true
        };

        this.servicioSeleccionado = null;
        this.fechaOcupada = false;
        this.cargarFechasOcupadas();
      },
      error: err => {
        this.mensajeModal =
          err.error?.message ||
          err.error ||
          'Error al registrar la reserva';
      }
    });
  }

  cerrarModal(): void {
    this.mensajeModal = '';
  }
}