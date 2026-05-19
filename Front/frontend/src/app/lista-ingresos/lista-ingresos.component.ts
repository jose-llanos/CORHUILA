import { CommonModule } from "@angular/common";
import { Component, OnInit } from "@angular/core";
import { RegistroIngresoVehiculoService } from "../registro-ingreso-vehiculo/registro-ingreso-vehiculo.service";
import { FormsModule } from "@angular/forms";

@Component({
    selector: "app-lista-ingresos",
    standalone: true,
    imports: [CommonModule, FormsModule],
    templateUrl: "./lista-ingresos.component.html",
    styleUrls: ["./lista-ingresos.component.css"],
})
export class ListaIngresosComponent implements OnInit {
    ingresos: any[] = [];
    cargando: boolean = true;
    ingresoSeleccionado: any = null;
    metodoPago: string = '';
    mostrarError: boolean = false;
    valorEfectivo: string = '';

    constructor(private registroService: RegistroIngresoVehiculoService) { }

    ngOnInit() {
        this.registroService.obtenerIngresos().subscribe({
            next: (data) => {
                this.ingresos = data;
                this.cargando = false;
            },
            error: () => {
                this.cargando = false;
            }
        });
    }

    mostrarFormularioPago(ingreso: any) {
        this.ingresoSeleccionado = ingreso;
        this.metodoPago = '';
        this.mostrarError = false;
    }

    cancelarPago() {
        this.ingresoSeleccionado = null;
        this.metodoPago = '';
        this.mostrarError = false;
    }

    seleccionarMetodoPago(metodo: string) {
        this.metodoPago = metodo;
    }

    onPagar(form: any) {
        if (form.invalid) {
            this.mostrarError = true;
            return;
        }
        console.log('Ingreso a eliminar:', this.ingresoSeleccionado);
        this.registroService.eliminarIngreso(this.ingresoSeleccionado.id).subscribe({
            next: () => {
                this.ingresos = this.ingresos.filter(i => i.id !== this.ingresoSeleccionado.id);
                this.ingresoSeleccionado = null;
                this.metodoPago = '';
                this.mostrarError = false;
            },
            error: (err) => {
                console.error('Error eliminando ingreso:', err);
                this.mostrarError = true;
            }
        });
    }

    formatearValorEfectivo(valor: string) {
        // Elimina todo lo que no sea número
        let soloNumeros = valor.replace(/\D/g, '');
        // Formatea con puntos de miles
        let conPuntos = soloNumeros.replace(/\B(?=(\d{3})+(?!\d))/g, '.');
        this.valorEfectivo = conPuntos;
    }

}