import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Producto } from './producto';  // Asegúrate de que la clase Producto esté importada
import { Carrito } from '../carrito/carrito'; // Asegúrate de importar Carrito

@Injectable({
  providedIn: 'root'
})
export class ProductoService {

  private urlEndpoint = 'http://localhost:8080/api/producto/';  // URL de los productos
  private carritoUrl = 'http://localhost:8080/api/carrito';  // URL para el carrito
  private httpHeaders = new HttpHeaders({'Content-Type': 'application/json'});

  constructor(private http: HttpClient) { }

  // Obtener todos los productos
  getProductos(): Observable<Producto[]> {
    return this.http.get<Producto[]>(this.urlEndpoint);
  }

  // Obtener un solo producto por ID
  getProducto(id: number): Observable<Producto> {
    return this.http.get<Producto>(`${this.urlEndpoint}/${id}`);
  }

  // Crear un producto
  create(producto: Producto): Observable<Producto> {
    return this.http.post<Producto>(this.urlEndpoint, producto, { headers: this.httpHeaders });
  }

  // Actualizar un producto
  update(producto: Producto): Observable<Producto> {
    return this.http.put<Producto>(`${this.urlEndpoint}/${producto.id}`, producto, { headers: this.httpHeaders });
  }

  // Eliminar un producto
  delete(id: number): Observable<void> {
    return this.http.delete<void>(`${this.urlEndpoint}/${id}`, { headers: this.httpHeaders });
  }

  // Método para agregar al carrito
  addToCart(carritoItem: Carrito): Observable<any> {
    return this.http.post<any>(this.carritoUrl, carritoItem, { headers: this.httpHeaders });
  }
}
