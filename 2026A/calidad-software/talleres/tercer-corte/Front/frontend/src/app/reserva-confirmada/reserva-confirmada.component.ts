import { CommonModule } from "@angular/common";
import { Component, OnInit } from "@angular/core";
import { ActivatedRoute, RouterModule } from "@angular/router";
import { ReservaService } from "../reserva/reservas.service";
import { Reserva } from "../reserva/reservas";
import { FormsModule } from "@angular/forms";

@Component({
  selector: 'app-reserva-confirmada',
  standalone: true,
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './reserva-confirmada.component.html',
  styleUrls: ['./reserva-confirmada.component.css']
})
export class ReservaConfirmadaComponent implements OnInit {
  reservas: Reserva[] = []; // Lista de reservas confirmadas
  isLoading: boolean = true; // Indicador de carga
  errorMessage: string = ''; // Mensaje de error

  reservaSeleccionada: Reserva | null = null; // Para el formulario de pago
  metodoPago: string = ''; // Método de pago seleccionado

  constructor(private reservaService: ReservaService) { }

  ngOnInit(): void {
    this.cargarReservasConfirmadas();
  }

  cargarReservasConfirmadas(): void {
    this.reservaService.getReservas().subscribe({
      next: (reservas) => {
        // Filtrar solo las reservas confirmadas
        this.reservas = reservas.filter((reserva) => reserva.confirmada);
        this.isLoading = false;
      },
      error: (err) => {
        this.errorMessage = 'Error al cargar las reservas confirmadas.';
        console.error(err);
        this.isLoading = false;
      }
    });
  }

  mostrarFormularioPago(reserva: Reserva) {
    this.reservaSeleccionada = reserva;
    this.metodoPago = '';
  }

  cancelarPago() {
    this.reservaSeleccionada = null;
    this.metodoPago = '';
  }

  seleccionarMetodoPago(metodo: string) {
    this.metodoPago = metodo;
  }

  mostrarError: boolean = false;

  onPagar(form: any) {
    if (form.invalid) {
      this.mostrarError = true;
      return;
    }
    this.mostrarError = false;
    this.pagarReserva();
  }

  numeroTarjeta: string = '';

  formatearNumeroTarjeta() {
    // Elimina todo lo que no sea dígito
    let valor = this.numeroTarjeta.replace(/\D/g, '');
    // Limita a 19 dígitos
    valor = valor.substring(0, 19);
    // Inserta un guion cada 4 dígitos
    let formateado = valor.replace(/(.{4})/g, '$1-');
    // Elimina el guion final si existe
    if (formateado.endsWith('-')) {
      formateado = formateado.slice(0, -1);
    }
    this.numeroTarjeta = formateado;
  }

  pagarReserva() {
  // Elimina la reserva pagada del array de reservas
  if (this.reservaSeleccionada) {
    this.reservas = this.reservas.filter(r => r !== this.reservaSeleccionada);
  }
  alert('Pago realizado con éxito');
  this.reservaSeleccionada = null;
  this.metodoPago = '';
  }

}