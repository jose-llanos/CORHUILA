import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {Tarifa} from './Tarifa'; // Asegúrate de que la ruta sea correcta
import { TarifaService } from './tarifa.service'; // Asegúrate de que la ruta sea correcta


@Component({
  selector: 'app-tarifa-admin',
  standalone: true,
  imports: [CommonModule, FormsModule], // Removed TarifaService from here
  templateUrl: './tarifa-admin.component.html',
  styleUrls: ['./tarifa-admin.component.css']
})
export class TarifaAdminComponent implements OnInit {
  tarifas: Tarifa[] = [];
  mostrarFormulario = false;
  tarifaSeleccionada: Tarifa = this.nuevaTarifa();

  constructor(private tarifaService: TarifaService) {}

  ngOnInit(): void {
    this.cargarTarifas();
  }

  cargarTarifas(): void {
    this.tarifaService.getTarifas().subscribe(data => {
      this.tarifas = data;
    });
  }

  abrirFormulario(): void {
    this.tarifaSeleccionada = this.nuevaTarifa();
    this.mostrarFormulario = true;
  }

  editarTarifa(tarifa: Tarifa): void {
    this.tarifaSeleccionada = { ...tarifa };
    this.mostrarFormulario = true;
  }

  guardarTarifa(): void {
    console.log('Datos enviados al backend:', this.tarifaSeleccionada);
  
    if (this.tarifaSeleccionada.id) {
      // Actualizar tarifa existente
      this.tarifaService.actualizarTarifa(this.tarifaSeleccionada.id, this.tarifaSeleccionada).subscribe({
        next: () => {
          this.cargarTarifas();
          this.mostrarFormulario = false;
        },
        error: (err) => {
          console.error('Error al actualizar la tarifa:', err);
        }
      });
    } else {
      // Crear nueva tarifa
      this.tarifaService.crearTarifa(this.tarifaSeleccionada).subscribe({
        next: (nuevaTarifa) => {
          this.tarifas.push(nuevaTarifa);
          this.mostrarFormulario = false;
        },
        error: (err) => {
          console.error('Error al crear la tarifa:', err);
        }
      });
    }
  }

  eliminarTarifa(id: number): void {
    if (confirm('¿Estás seguro de eliminar esta tarifa?')) {
      this.tarifaService.eliminarTarifa(id).subscribe(() => {
        this.cargarTarifas();
      });
    }
  }

  cancelar(): void {
    this.mostrarFormulario = false;
    this.tarifaSeleccionada = this.nuevaTarifa();
  }

  nuevaTarifa(): Tarifa {
    return {
      id: undefined, // explícitamente sin valor
      tipoVehiculo: '',
      tarifaDiurna: 0,
      tarifaNocturna: 0,
      imagen: ''
    };
  }
  
}
