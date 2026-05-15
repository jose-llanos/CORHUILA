import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { reserva } from './Reserva';
import { Services } from '../services/Service';

@Injectable({
  providedIn: 'root'
})
export class ReservesServiceService {
  private urlEndpoint: string = 'http://localhost:8080/autospark/reserva';
  private tiposServicioEndpoint: string = 'http://localhost:8080/autospark/service';
  private httpHeaders = new HttpHeaders({ 'Content-Type': 'application/json' });

  constructor(private http: HttpClient) {}

  getTiposServicio(): Observable<Services[]> {
    return this.http.get<Services[]>(this.tiposServicioEndpoint, {
      headers: this.httpHeaders
    });
  }

  getFechasOcupadas(): Observable<string[]> {
    return this.http.get<string[]>(`${this.urlEndpoint}/fechas-ocupadas`, {
      headers: this.httpHeaders
    });
  }

  create(reserva: reserva): Observable<reserva> {
    console.log('Enviando reserva al backend:', reserva);

    return this.http.post<reserva>(this.urlEndpoint, reserva, {
      headers: this.httpHeaders
    });
  }
}