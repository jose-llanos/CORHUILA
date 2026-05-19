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
    reservationTime: '',
    active: true
  };

  mensajeModal: string = '';
  tiposServicio: Services[] = [];
  servicioSeleccionado: Services | null = null;

  fechasOcupadas: string[] = [];
  fechaOcupada: boolean = false;
  fechaMinimaDate: string = '';

  constructor(private reservasService: ReservesServiceService) {}

  ngOnInit(): void {
    this.fechaMinimaDate = this.obtenerFechaMinimaDate();
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
        this.fechasOcupadas = fechas;
        console.log('Fechas ocupadas del backend:', this.fechasOcupadas);
      },
      error: (err) => {
        console.error('Error al obtener fechas ocupadas', err);
      }
    });
  }

  obtenerFechaMinimaDate(): string {
    const ahora = new Date();
    ahora.setDate(ahora.getDate() + 1);
    return ahora.toISOString().split('T')[0];
  }

  actualizarPrecio(): void {
    if (this.servicioSeleccionado) {
      this.reservaForm.serviceType = this.servicioSeleccionado.name;
      this.reservaForm.value = this.servicioSeleccionado.price;
    }
  }

  validarFechaSeleccionada(): void {
    const fecha = this.reservaForm.reservationDate;
    const hora = this.reservaForm.reservationTime;
    
    if (!fecha || !hora) {
      this.fechaOcupada = false;
      return;
    }
    
    const fechaCompleta = `${fecha}T${hora}`;
    this.fechaOcupada = this.fechasOcupadas.includes(fechaCompleta);
    
    if (this.fechaOcupada) {
      this.mensajeModal = 'Ese horario ya está ocupado. Por favor selecciona otra fecha u hora.';
    }
  }

  registrarReserva(): void {
    // Validaciones adicionales
    if (!this.reservaForm.reservationDate || !this.reservaForm.reservationTime) {
      this.mensajeModal = 'Por favor selecciona fecha y hora para la reserva.';
      return;
    }
    
    this.validarFechaSeleccionada();

    if (this.fechaOcupada) {
      this.mensajeModal = 'No puedes registrar una reserva en un horario ocupado.';
      return;
    }

    // ✅ CORREGIDO: Formatear la fecha correctamente para el backend
    // El backend espera: { "reservationDate": "2026-12-25", "reservationTime": "15:00:00" }
    const reservaToSend = {
      vehicleType: this.reservaForm.vehicleType,
      serviceType: this.reservaForm.serviceType,
      licensePlate: this.reservaForm.licensePlate,
      value: this.reservaForm.value,
      reservationDate: this.reservaForm.reservationDate,  // Formato YYYY-MM-DD
      reservationTime: this.reservaForm.reservationTime   // Formato HH:MM:SS
    };

    console.log('Enviando reserva al backend:', reservaToSend);

    this.reservasService.create(reservaToSend).subscribe({
      next: (response) => {
        console.log('Respuesta exitosa del servidor:', response);
        // ✅ Mostrar mensaje de éxito
        this.mensajeModal = '✅ ¡Reserva registrada exitosamente!';
        
        // Resetear formulario
        this.resetForm();
        
        // Recargar fechas ocupadas
        this.cargarFechasOcupadas();
      },
      error: (err) => {
        console.error('Error completo:', err);
        
        let mensajeError = '❌ Error al registrar la reserva';
        
        if (err.error) {
          // Si el error es string
          if (typeof err.error === 'string') {
            mensajeError = err.error;
          }
          // Si tiene mensaje
          else if (err.error.message) {
            mensajeError = err.error.message;
          }
          // Si es el error de conflicto del backend
          else if (err.status === 409) {
            mensajeError = '⚠️ Este horario ya está ocupado. Por favor selecciona otro.';
          }
        }
        
        this.mensajeModal = mensajeError;
      }
    });
  }

  resetForm(): void {
    this.reservaForm = {
      id: 0,
      vehicleType: 'CARRO',
      serviceType: '',
      licensePlate: '',
      value: 0,
      reservationDate: '',
      reservationTime: '',
      active: true
    };
    this.servicioSeleccionado = null;
    this.fechaOcupada = false;
  }

  cerrarModal(): void {
    this.mensajeModal = '';
  }
}