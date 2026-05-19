import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
import { User } from './user';
import { Router } from '@angular/router';

@Injectable({ providedIn: 'root' })
export class LoginService {
  private urlEndPoint: string = 'http://localhost:8080/api/user';
  private httpHeaders = new HttpHeaders({ 'Content-Type': 'application/json' });

  constructor(private http: HttpClient, private router: Router) {}

  // Obtener todos los usuarios registrados
  getUsers(): Observable<User[]> {
    return this.http.get<User[]>(this.urlEndPoint).pipe(
      map(response => response as User[]),
      catchError(() => of([])) // Manejo de error: si falla, devuelve un array vacío
    );
  }

  // Método para encriptar la contraseña con SHA-256
  async encryptPassword(password: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(password);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    return Array.from(new Uint8Array(hashBuffer))
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');
  }

  // Validar usuario con la contraseña encriptada
  async validateUser(email: string, password: string): Promise<User | null> {
    const encryptedPassword = await this.encryptPassword(password);
    return this.getUsers().pipe(
      map(users => {
        const user = users.find(u => u.email === email && u.password === encryptedPassword);
        if (user) {
          console.log('Usuario encontrado:', user); // <- Este log te ayudará a depurar
        } else {
          console.warn('No se encontró usuario con ese email/contraseña');
        }
        return user || null;
      })
    ).toPromise();
  }

  // Redirigir según el rol
  redirectByRole(role: string): void {
    if (role === 'Administrador') {
      // Redirigir al panel de administración
      window.location.href = '/admin';
    } else if (role === 'cliente') {
      // Redirigir al panel de cliente
      window.location.href = '/login/view-cliente';
    } else {
      // Redirigir a una página genérica o de error
      window.location.href = '/error';
    }
  }
}
