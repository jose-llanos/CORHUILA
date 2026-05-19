export class Inventario {
    id: number;
    productoId: number;
    cantidad: number;
  
    constructor(id: number = 0, productoId: number = 0, cantidad: number = 0) {
      this.id = id;
      this.productoId = productoId;
      this.cantidad = cantidad;
    }
  }
  