import { Component, OnInit } from '@angular/core';
import { Producto} from '../producto/producto';
import { ProductoService } from '../producto/producto.service';
import { Router, RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common'; // Importa CommonModule para ngFor, ngIf, etc.

@Component({
  selector: 'app-inventario',
  standalone: true,
  imports: [CommonModule, RouterModule], // Asegúrate de importar CommonModule
  templateUrl: './inventario.component.html',
  styleUrls: ['./inventario.component.css']
})
export class InventarioComponent implements OnInit {
  productos: Producto[] = [];

  constructor(private productoService: ProductoService, private router: Router) { }

  ngOnInit(): void {
    this.productoService.getProductos().subscribe(productos => this.productos = productos);
  }

  eliminarProducto(producto: Producto): void {
    this.productoService.delete(producto.id).subscribe(() => {
      this.productos = this.productos.filter(p => p.id !== producto.id);
    });
  }
}
