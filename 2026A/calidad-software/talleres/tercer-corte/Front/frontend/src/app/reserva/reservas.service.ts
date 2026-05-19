import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { map, Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { Reserva } from './reservas';

@Injectable({
  providedIn: 'root'
})
export class ReservaService {

  private urlEndpoint: string = "http://localhost:8080/api/reservas";
  private httpHeaders = new HttpHeaders({ 'Content-Type': 'application/json' });

  constructor(private http: HttpClient) {}

  create_reservation(reserva: Reserva): Observable<Reserva> {
    return this.http.post<Reserva>(`${this.urlEndpoint}`, reserva, {
      headers: this.httpHeaders
    }).pipe(
      catchError(this.handleError)
    );
  }

    update_reservation(id: number, reserva: Reserva): Observable<Reserva> {
    console.log('Datos enviados para actualizar:', reserva); // Depuración
    return this.http.put<Reserva>(`${this.urlEndpoint}/${id}`, reserva, {
      headers: this.httpHeaders
      }).pipe(
      catchError(this.handleError)
      );
    }

    getReservas(): Observable<Reserva[]> {
      return this.http.get<Reserva[]>(this.urlEndpoint).pipe(
        map((response) => response as Reserva[]),
        catchError(this.handleError)
      );
    }

    obtenerReserva(id: number): Observable<Reserva> {
      return this.http.get<Reserva>(`${this.urlEndpoint}/${id}`, {
        headers: this.httpHeaders
      }).pipe(
        catchError(this.handleError)
      );
    }

    guardarReserva(reserva: Reserva): Observable<Reserva> {
      return this.http.post<Reserva>(this.urlEndpoint, reserva, {
        headers: this.httpHeaders
      }).pipe(
        catchError(this.handleError)
      );
    }

    confirmarReserva(id: number): Observable<any> {
      return this.http.put(`${this.urlEndpoint}/${id}/confirmar`, null, {
        headers: this.httpHeaders
      }).pipe(
        catchError(this.handleError)
      );
    }

    private handleError(error: any): Observable<never> {
      console.error('Error en la solicitud HTTP:', error);
      return throwError(() => new Error('Error en la solicitud HTTP'));
    }


    deleteReserva(id: number): Observable<void> {
    return this.http.delete<void>(`${this.urlEndpoint}/${id}`).pipe(
      catchError(this.handleError)
    );
  }
}
