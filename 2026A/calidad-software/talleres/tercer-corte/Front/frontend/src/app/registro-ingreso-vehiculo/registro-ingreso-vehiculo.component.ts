import { CommonModule } from "@angular/common";
import { Component } from "@angular/core";
import { FormsModule } from "@angular/forms";
import { RegistroIngresoVehiculoService } from "./registro-ingreso-vehiculo.service";


@Component({
    selector: "app-registro-ingreso-vehiculo",
    standalone: true,
    imports: [CommonModule, FormsModule],
    templateUrl: "./registro-ingreso-vehiculo.component.html",
    styleUrls: ["./registro-ingreso-vehiculo.component.css"],
})
export class RegistroIngresoVehiculoComponent {
    placa: string = "";
    tipoVehiculo: string = "";
    horaIngreso: string = "";
    mensaje: string = "";
    ubicacion: string = '';
    registroExitoso: boolean = false;

    constructor(private registroService: RegistroIngresoVehiculoService) {}

    registrarIngreso(form: any) {
        if (form.invalid) {
            this.registroExitoso = false;
            return;
        }
        const ingreso = {
            placa: this.placa,
            tipoVehiculo: this.tipoVehiculo,
            ubicacion: this.ubicacion,
            horaIngreso: this.horaIngreso, 
        };
       this.registroService.registrarIngreso(ingreso).subscribe({
            next: () => {
                this.registroExitoso = true;
                form.resetForm();
                setTimeout(() => this.registroExitoso = false, 2000);
            },
            error: () => {
                this.registroExitoso = false;
            }
        });
    }
}