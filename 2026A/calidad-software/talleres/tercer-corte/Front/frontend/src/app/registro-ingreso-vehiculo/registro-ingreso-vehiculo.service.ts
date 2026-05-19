import { HttpClient } from "@angular/common/http";
import { Injectable } from "@angular/core";
import { Observable } from "rxjs";

@Injectable({
    providedIn: 'root'
})
export class RegistroIngresoVehiculoService {
    private apiUrl = 'http://localhost:8080/api/ingresos';

    constructor(private http: HttpClient) {}

    registrarIngreso(ingreso: any): Observable<any> {
        return this.http.post(this.apiUrl, ingreso);
    }

    obtenerIngresos(): Observable<any[]> {
    return this.http.get<any[]>(this.apiUrl);
    }
    
    eliminarIngreso(id: number): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/${id}`);
}

}