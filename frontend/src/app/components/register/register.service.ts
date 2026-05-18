import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { Usuario } from './usuario';


@Injectable({
  providedIn: 'root'
})
export class RegisterService {
  private urlEndpoint: string = "http://localhost:8080/autospark/users";
  private httpHeaders = new HttpHeaders({ 'Content-Type': 'application/json' });

  constructor(private http: HttpClient) {}

  create(usuario: Usuario): Observable<Usuario> {
    return this.http.post<Usuario>(this.urlEndpoint, usuario, { headers: this.httpHeaders });
  }
}