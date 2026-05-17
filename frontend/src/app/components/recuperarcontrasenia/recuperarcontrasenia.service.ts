import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class RecuperarContraseniaService {
  private urlEndpoint: string = "http://host.docker.internal:8080/autospark/recover-password";

  constructor(private http: HttpClient) {}

  recuperarContrasenia(email: string): Observable<string> {
    const params = new HttpParams().set('email', email);
    return this.http.post<string>(this.urlEndpoint, null, { params, responseType: 'text' as 'json' });
  }
}