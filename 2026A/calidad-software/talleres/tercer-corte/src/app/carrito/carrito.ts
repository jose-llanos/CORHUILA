import { Producto } from '../producto/producto';  // Asegúrate de importar la clase Producto

export class Carrito {
  id: number = 0;
  producto: Producto;  // El carrito tiene un producto
  cantidad: number;  // Cantidad de ese producto en el carrito
  total: number;  // Total de ese producto (cantidad * precio)

  constructor(producto: Producto, cantidad: number) {
    this.producto = producto;
    this.cantidad = cantidad;
    this.total = producto.precio * cantidad;  // Calculamos el total con el precio y la cantidad
  }
}
