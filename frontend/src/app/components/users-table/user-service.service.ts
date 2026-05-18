import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

import { UsersTable } from './UsersTable';

@Injectable({
  providedIn: 'root'
})
export class UserServiceService {

  private urlEndpoint: string = '/autospark/users';

  private httpHeaders = new HttpHeaders({
    'Content-Type': 'application/json'
  });

  constructor(private http: HttpClient) {}

  getUsers(): Observable<UsersTable[]> {
    return this.http.get<UsersTable[]>(this.urlEndpoint, {
      headers: this.httpHeaders
    });
  }

 changeRole(id: number, role: 'CUSTOMER' | 'ADMIN'): Observable<any> {
  return this.http.put(
    `${this.urlEndpoint}/${id}/role?role=${role}`,
    {},
    {
      headers: this.httpHeaders,
      responseType: 'text' as 'json'
    }
  );
}
}