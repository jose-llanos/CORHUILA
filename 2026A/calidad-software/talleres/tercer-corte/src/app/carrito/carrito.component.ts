import { Component, OnInit } from '@angular/core';
import { CarritoService } from './carrito.service';  // Servicio para manejar el carrito
import { Carrito } from './carrito';  // La clase de Carrito
import { CommonModule } from '@angular/common';  // Para usar ngIf, ngFor, etc.
import { FormsModule } from '@angular/forms';  // Para usar ngModel
import Swal from 'sweetalert2';

@Component({
  selector: 'app-carrito',
  standalone: true,  // Marcar el componente como standalone
  imports: [CommonModule, FormsModule],  // Importar los módulos necesarios
  templateUrl: './carrito.component.html',
  styleUrls: ['./carrito.component.css']
})
export class CarritoComponent implements OnInit {
  carritoItems: Carrito[] = [];  // Arreglo de productos en el carrito
  total: number = 0;

  constructor(private carritoService: CarritoService) {}

  ngOnInit(): void {
    this.cargarCarrito();
  }

  cargarCarrito(): void {
    this.carritoService.getCarrito().subscribe(
      (data: Carrito[]) => {
        this.carritoItems = data;
        this.calcularTotal();
      },
      (error) => {
        Swal.fire('Error', 'Hubo un problema al cargar el carrito', 'error');
      }
    );
  }

  actualizarCantidad(item: Carrito): void {
    console.log('Actualizando cantidad para item con id:', item.id, 'nueva cantidad:', item.cantidad);
    
    item.total = item.producto.precio * item.cantidad;  // Actualiza total de cada producto
    this.carritoService.updateQuantity(item.id, item.cantidad).subscribe(
      (response) => {
        console.log('Respuesta del servidor:', response);
        this.calcularTotal();
        Swal.fire('Cantidad actualizada', '', 'success');
      },
      (error) => {
        console.error('Error al actualizar la cantidad:', error);
        Swal.fire('Error', 'No se pudo actualizar la cantidad', 'error');
      }
    );
  }

  eliminarProducto(item: Carrito): void {
    this.carritoService.deleteFromCart(item.id).subscribe(
      (response) => {
        this.carritoItems = this.carritoItems.filter(i => i.id !== item.id);  // Eliminar de la lista
        this.calcularTotal();
        Swal.fire('Producto eliminado', '', 'success');
      },
      (error) => {
        Swal.fire('Error', 'No se pudo eliminar el producto', 'error');
      }
    );
  }

  vaciarCarrito(): void {
    this.carritoService.vaciarCarrito().subscribe(
      (response) => {
        this.carritoItems = [];
        this.calcularTotal();
        Swal.fire('Carrito vacio', '', 'success');
      },
      (error) => {
        Swal.fire('Error', 'No se pudo vaciar el carrito', 'error');
      }
    );
  }

  calcularTotal(): number {
    this.total = this.carritoItems.reduce((acc, item) => acc + item.total, 0);  // Sumar el total de todos los productos
    return this.total;
  }
}
