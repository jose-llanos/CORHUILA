import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { ProductoService } from './producto.service';
import { Producto } from './producto';
import { Location } from '@angular/common';
import { FormsModule } from '@angular/forms'; // Importa FormsModule para ngModel

@Component({
  selector: 'app-form',
  standalone: true,
  imports: [FormsModule],  // Asegúrate de importar FormsModule para poder usar ngModel
  templateUrl: './productoform.component.html',
})
export class FormComponentProducto implements OnInit {
  producto: Producto = { id: 0, nombre: '', descripcion: '', precio: 0, cantidad: 0, imagenUrl: '' };
  titulo: string = 'Crear Producto';

  constructor(
    private productoService: ProductoService,
    private route: ActivatedRoute,
    private router: Router,
    private location: Location
  ) {}

  ngOnInit(): void {
    const id = this.route.snapshot.paramMap.get('id');
    if (id) {
      this.titulo = 'Editar Producto';
      this.productoService.getProducto(Number(id)).subscribe((producto) => {
        this.producto = producto;
      });
    }
  }

  save(): void {
    // Aquí conectamos la ruta base para las imágenes
    this.producto.imagenUrl = `assets/productos/${this.producto.imagenUrl}`;

    if (this.producto.id === 0) {
      this.productoService.create(this.producto).subscribe(() => {
        this.router.navigate(['/inventario']);
      });
    } else {
      this.productoService.update(this.producto).subscribe(() => {
        this.router.navigate(['/inventario']);
      });
    }
  }

  cancel(): void {
    this.location.back();
  }
}
