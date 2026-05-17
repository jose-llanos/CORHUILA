import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { ReservaUsuario } from './ReservaUsuario';

@Injectable({
  providedIn: 'root'
})
export class TablaServiceService {
  private urlEndpoint: string = "http://host.docker.internal:8080/autospark/reservas-con-usuarios";
  private httpHeaders = new HttpHeaders({ 'Content-Type': 'application/json' });
  constructor(private http: HttpClient) { }

  getReservasConUsuario(): Observable<ReservaUsuario[]> {
    return this.http.get<ReservaUsuario[]>(this.urlEndpoint, { headers: this.httpHeaders });
  }
}
