import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { User } from '../login_register/user';
import { MapServices } from './admin';

@Injectable({ providedIn: 'root' })

export class LoginService {
  private urlEndPoint: string = 'http://localhost:8080/api/user';
  private httpHeaders = new HttpHeaders({ 'Content-Type': 'application/json' });

  constructor(private http: HttpClient) {}

  // Obtener todos los usuarios registrados
  getUsers(): Observable<User[]> {
    return this.http.get<User[]>(this.urlEndPoint).pipe(
      map(response => response as User[])
    );
  }


}

@Injectable({ providedIn: 'root' })
export class AdminService {
  private urlEndPointtwo = 'http://localhost:8080/api/reservas'; // URL del backend

  constructor(private http: HttpClient) {}

  getServices(): Observable<MapServices[]> {
    return this.http.get<MapServices[]>(this.urlEndPointtwo);
  }

  createService(service: MapServices): Observable<MapServices> {
    return this.http.post<MapServices>(this.urlEndPointtwo, service);
  }

  updateService(id: number, service: MapServices): Observable<MapServices> {
    return this.http.put<MapServices>(`${this.urlEndPointtwo}/${id}`, service);
  }

  deleteService(id: number): Observable<void> {
    return this.http.delete<void>(`${this.urlEndPointtwo}/${id}`);
  }
}