import { Component, OnInit } from '@angular/core';
import { ProductoService } from './producto.service';
import { CarritoService } from '../carrito/carrito.service'; // Servicio para manejar el carrito
import { Producto } from './producto';
import { Carrito } from '../carrito/carrito'; // Asegúrate de importar Carrito
import Swal from 'sweetalert2';
import { CommonModule } from '@angular/common'; // Importa CommonModule para ngFor, ngIf
import { FormsModule } from '@angular/forms'; // Importar FormsModule para ngModel
import { RouterModule } from '@angular/router'; // Importa RouterModule si utilizas el enrutador

@Component({
  selector: 'app-producto',
  standalone: true,  // Marca el componente como standalone
  imports: [CommonModule, FormsModule, RouterModule],  // Asegúrate de importar CommonModule, FormsModule, RouterModule
  templateUrl: './producto.component.html',
  styleUrls: ['./producto.component.css']
})
export class ProductoComponent implements OnInit {

  productos: Producto[] = [];

  constructor(
    private productoService: ProductoService,
    private carritoService: CarritoService,  // Servicio para manejar el carrito
  ) { }

  ngOnInit(): void {
    this.productoService.getProductos().subscribe(
      productos => {
        // Inicializa la cantidad de cada producto en 1 por defecto
        this.productos = productos.map(producto => ({
          ...producto,
          cantidad: 1  // Cada producto tendrá su cantidad por defecto de 1
        }));
      }
    );
  }
  
  addToCart(producto: Producto): void {
    // Validar que la cantidad sea válida (mayor que 0)
    if (producto.cantidad <= 0 || isNaN(producto.cantidad)) {
      Swal.fire('Error', 'La cantidad debe ser un número mayor a cero', 'error');
      return;
    }
  
    Swal.fire({
      title: "Producto agregado",
      text: `¿Quieres agregar ${producto.cantidad} unidades del producto ${producto.nombre} al carrito?`,
      icon: "success",
      showCancelButton: true,
      confirmButtonText: "Sí, agregar",
      cancelButtonText: "Cancelar"
    }).then((result) => {
      if (result.isConfirmed) {
        // Crear el objeto Carrito con el producto y la cantidad seleccionada
        const carritoItem: Carrito = new Carrito(producto, producto.cantidad);  // Ahora se pasan los argumentos correctamente
  
        // Llamar al servicio para agregar el producto al carrito
        this.carritoService.addToCart(carritoItem).subscribe(
          response => {
            Swal.fire('Producto agregado al carrito', '', 'success');
            producto.cantidad = 1;  // Resetear la cantidad del producto después de agregarlo al carrito
          },
          error => {
            Swal.fire('Error', 'Hubo un problema al agregar el producto al carrito', 'error');
          }
        );
      }
    });
  }
}
