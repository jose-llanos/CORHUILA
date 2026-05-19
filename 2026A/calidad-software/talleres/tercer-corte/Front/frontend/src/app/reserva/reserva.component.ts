import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { ReservaService } from './reservas.service';
import { HttpClient } from '@angular/common/http';
import { Reserva } from './reservas';

@Component({
  selector: 'app-reserva',
  standalone: true,
  imports: [RouterModule, FormsModule, CommonModule],
  templateUrl: './reserva.component.html',
  styleUrls: ['./reserva.component.css']
})
export class ReservaComponent implements OnInit {
  // Variables de estado
  editando: boolean = false;
  isLoading: boolean = true;
  errorMessage: string = '';

  // Listas de opciones
  tipo_servicio: string[] = ['Mensualidad', 'Servicio Nocturno', 'Techado', 'Valet Parking'];
  tipo_vehiculo: string[] = ['Carro', 'Moto', 'Camioneta'];

  // Objeto de reserva actual
  reserved: Reserva = {
    id: 0,
    tipo_vehiculo: '',
    tipo_servicio: '',
    horas: 0,
    fecha: '',
    precio: null,
    confirmada: false
  };

  // Lista de reservas
  reservas: Reserva[] = [];

  // Precios por combinación de vehículo y servicio
  private precios: { [key: string]: number } = {
    'Carro-Mensualidad': 300000,
    'Carro-Servicio Nocturno': 15000,
    'Carro-Techado': 25000,
    'Carro-Valet Parking': 50000,
    'Moto-Mensualidad': 200000,
    'Moto-Servicio Nocturno': 12000,
    'Moto-Techado': 20000,
    'Moto-Valet Parking': 40000,
    'Camioneta-Mensualidad': 350000,
    'Camioneta-Servicio Nocturno': 18000,
    'Camioneta-Techado': 28000,
    'Camioneta-Valet Parking': 55000
  };

  constructor(
    private reservaService: ReservaService,
    private http: HttpClient,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.loadReservas();
  }

  // Métodos principales
  loadReservas(): void {
    this.isLoading = true;
    this.reservaService.getReservas().subscribe({
      next: (data) => {
        this.reservas = data;
        this.isLoading = false;
      },
      error: (err) => {
        this.handleError('No se pudieron cargar las reservas.', err);
      }
    });
  }

  registrarReserva(): void {
    if (!this.validarReserva()) return;

    this.reservaService.create_reservation(this.reserved).subscribe({
      next: () => {
        alert('Reserva registrada correctamente.');
        this.limpiarFormulario();
        this.loadReservas();
      },
      error: (err) => {
        this.handleError('Error al registrar la reserva.', err);
      }
    });
  }

 actualizarReserva(): void {
  this.reservaService.update_reservation(this.reserved.id, this.reserved).subscribe({
    next: (response) => {
      console.log('Respuesta del backend:', response);
      alert('Reserva actualizada correctamente.');
      this.limpiarFormulario();
      this.loadReservas();
    },
    error: (err) => {
      this.handleError('Error al actualizar la reserva.', err);
    }
  });
}

  editarReserva(reserva: Reserva): void {
    this.editando = true;
    this.reserved = { ...reserva }; // Copia los datos de la reserva seleccionada
    console.log('Reserva seleccionada para editar:', this.reserved);
    window.scrollTo({ top: 0, behavior: 'smooth' }); // Opcional: desplaza la vista hacia arriba
  }

  eliminarReserva(id: number): void {
    if (confirm('¿Estás seguro de que deseas eliminar esta reserva?')) {
      this.reservaService.deleteReserva(id).subscribe({
        next: () => {
          alert('Reserva eliminada correctamente.');
          this.loadReservas();
        },
        error: (err) => {
          this.handleError('Error al eliminar la reserva.', err);
        }
      });
    }
  }

  confirmarReserva(id: number): void {
    const reserva = this.reservas.find((r) => r.id === id);
    if (!reserva) {
      alert('Reserva no encontrada.');
      return;
    }

    reserva.confirmada = true;
    this.reservaService.update_reservation(id, reserva).subscribe({
      next: () => {
        alert('Reserva confirmada correctamente.');
        this.loadReservas();
        this.router.navigate(['/reservas-confirmadas']);
      },
      error: (err) => {
        this.handleError('Error al confirmar la reserva.', err);
      }
    });
  }

  // Métodos auxiliares
  limpiarFormulario(): void {
    this.reserved = {
      id: 0,
      tipo_vehiculo: '',
      tipo_servicio: '',
      horas: 0,
      fecha: '',
      precio: null,
      confirmada: false
    };
    this.editando = false; // Cambia el estado de edición a falso
  }

  actualizarPrecio(): void {
    const { tipo_vehiculo, tipo_servicio } = this.reserved;
    const key = `${tipo_vehiculo}-${tipo_servicio}`;
    this.reserved.precio = this.precios[key] || null;
  }

  private validarReserva(): boolean {
    const { tipo_vehiculo, tipo_servicio, precio } = this.reserved;

    if (!tipo_vehiculo || !tipo_servicio) {
      alert('Debes seleccionar un tipo de vehículo y un tipo de servicio.');
      return false;
    }

    if (precio === null || precio === 0) {
      alert('Esta combinación no está disponible. Por favor, elige otra.');
      return false;
    }

    return true;
  }

  private getUserId(): string | null {
    const userId = localStorage.getItem('userId');
    if (!userId) {
      this.handleMissingUserId();
    }
    return userId;
  }

  private handleMissingUserId(): void {
    this.errorMessage = 'No se pudo identificar al usuario. Por favor, inicie sesión nuevamente.';
    this.isLoading = false;
  }

  private handleError(message: string, error: any): void {
    console.error(message, error);
    alert(message);
    this.isLoading = false;
  }

  onChange(campo: string, valor: any): void {
  console.log(`Campo actualizado: ${campo}, Valor: ${valor}`);
}
}
