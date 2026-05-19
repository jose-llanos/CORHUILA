import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class RecoverpasswordService {
  private urlEndpoint: string = 'http://localhost:8080/api/recuperarcontrasenia';

  constructor(private http: HttpClient) {}

  recuperarContrasenia(email: string): Observable<string> {
    const params = new HttpParams().set('email', email);
    return this.http.post<string>(this.urlEndpoint, null, { params, responseType: 'text' as 'json' });
  }
}