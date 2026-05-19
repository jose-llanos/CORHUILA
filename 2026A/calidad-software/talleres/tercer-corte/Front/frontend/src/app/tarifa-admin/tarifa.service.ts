import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Tarifa } from './Tarifa'; 
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class TarifaService {
  private baseUrl = 'http://localhost:8080/api/tarifas';

  constructor(private http: HttpClient) {}

  getTarifas(): Observable<Tarifa[]> {
    return this.http.get<Tarifa[]>(this.baseUrl);
  }

  getTarifa(id: number): Observable<Tarifa> {
    return this.http.get<Tarifa>(`${this.baseUrl}/${id}`);
  }

  crearTarifa(tarifa: Tarifa): Observable<Tarifa> {
    return this.http.post<Tarifa>(this.baseUrl, tarifa);
  }

  actualizarTarifa(id: number, tarifa: Tarifa): Observable<Tarifa> {
    return this.http.put<Tarifa>(`${this.baseUrl}/${id}`, tarifa);
  }

  eliminarTarifa(id: number): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/${id}`);
  }
}
