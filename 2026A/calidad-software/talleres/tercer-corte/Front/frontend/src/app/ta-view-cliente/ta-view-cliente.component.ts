import { Component } from '@angular/core';
import { Tarifa } from '../tarifa-admin/Tarifa'; // Asegúrate de que la ruta sea correcta
import { TarifaService } from '../tarifa-admin/tarifa.service'; // Asegúrate de que la ruta sea correcta
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-ta-view-cliente',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './ta-view-cliente.component.html',
  styleUrl: './ta-view-cliente.component.css'
})
export class TaViewClienteComponent {


  tarifas: Tarifa[] = [];

  constructor(private tarifaService: TarifaService) {}

  ngOnInit(): void {
    this.obtenerTarifas();
  }

  obtenerTarifas(): void {
    this.tarifaService.getTarifas().subscribe((data) => {
      this.tarifas = data;
    });
  }

  eliminarTarifa(id: number): void {
    this.tarifaService.eliminarTarifa(id).subscribe(() => {
      this.obtenerTarifas(); // refrescar la lista
    });
  }

  editarTarifa(tarifa: Tarifa): void {
    // A implementar
    console.log('Editar:', tarifa);
  }

}
