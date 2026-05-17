// reserves-service.service.ts
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { reserva } from './Reserva';
import { Services } from '../services/Service';

@Injectable({
  providedIn: 'root'
})
export class ReservesServiceService {
  private urlEndpoint: string = 'http://host.docker.internal:8080/autospark/reserva';
  private tiposServicioEndpoint: string = 'http://host.docker.internal:8080/autospark/service';
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

  // ✅ CORREGIDO: No enviar el objeto reserva completo, sino un DTO específico
  create(reservaData: any): Observable<any> {
    // Construir el payload exactamente como lo espera el backend
    const payload = {
      vehicleType: reservaData.vehicleType,
      serviceType: reservaData.serviceType,
      licensePlate: reservaData.licensePlate,
      value: reservaData.value,
      reservationDate: reservaData.reservationDate,  // Formato: "2025-12-25"
      reservationTime: reservaData.reservationTime   // Formato: "15:00:00"
    };
    
    console.log('📤 Enviando payload al backend:', payload);
    console.log('📅 Fecha:', payload.reservationDate);
    console.log('⏰ Hora:', payload.reservationTime);
    
    return this.http.post(this.urlEndpoint, payload, {
      headers: this.httpHeaders
    });
  }
}