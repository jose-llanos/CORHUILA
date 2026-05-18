import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Carrito } from './carrito';
import { Producto } from '../producto/producto';

@Injectable({
  providedIn: 'root'
})
export class CarritoService {
  private carritoUrl = 'http://localhost:8080/api/carrito';  // URL para el carrito en el backend
  private httpHeaders = new HttpHeaders({ 'Content-Type': 'application/json' });

  constructor(private http: HttpClient) { }

  // Obtener los productos en el carrito desde el backend
  getCarrito(): Observable<Carrito[]> {
    return this.http.get<Carrito[]>(this.carritoUrl, { headers: this.httpHeaders });
  }

  // Agregar un producto al carrito
  addToCart(carritoItem: Carrito): Observable<Carrito> {
    // Verificar que carritoItem tenga los campos necesarios
    console.log('Producto enviado al carrito:', carritoItem);
    return this.http.post<Carrito>(this.carritoUrl, carritoItem, { headers: this.httpHeaders });
  }

  // Eliminar un producto del carrito
  deleteFromCart(carritoId: number): Observable<void> {
    return this.http.delete<void>(`${this.carritoUrl}/${carritoId}`, { headers: this.httpHeaders });
  }

  // Actualizar la cantidad de un producto en el carrito  
  updateQuantity(carritoId: number, cantidad: number): Observable<Carrito> {
  const url = `${this.carritoUrl}/${carritoId}?cantidad=${cantidad}`;
  return this.http.put<Carrito>(url, {}, { headers: this.httpHeaders });
  }

  // Vaciar todo el carrito
  vaciarCarrito(): Observable<void> {
    return this.http.delete<void>(this.carritoUrl, { headers: this.httpHeaders });
  }
}
