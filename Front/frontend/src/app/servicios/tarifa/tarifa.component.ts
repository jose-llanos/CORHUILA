import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-tarifa',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './tarifa.component.html',
  styleUrls: ['./tarifa.component.css']
})
export class TarifaComponent {
  vehiculo = {
    typecar: '',
    plate: '',
    hours: 0
  };

  resultado: number | null = null;
  errorMessage: string | null = null;

  constructor(private http: HttpClient) {}

  calcularTarifa(): void {
    const { typecar, plate, hours } = this.vehiculo;

    if (!typecar || !plate || hours <= 0) {
      alert('Por favor, completa todos los campos correctamente.');
      return;
    }

    // Enviar los datos al backend para validar y calcular la tarifa
    this.http.post<number>('http://localhost:8080/api/validar-tarifa', { typecar, plate, hours }).subscribe({
      next: (total) => {
        this.resultado = total;
        this.errorMessage = null; // Limpiar mensaje de error si todo está bien
      },
      error: (err) => {
        console.error('Error al calcular la tarifa:', err);
        this.errorMessage = 'No se pudo calcular la tarifa. Verifica los datos ingresados.';
      }
    });
  }
}
