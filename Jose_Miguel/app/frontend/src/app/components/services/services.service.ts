import { HttpClient, HttpHeaders } from "@angular/common/http";
import { Injectable } from "@angular/core";
import { Services } from './Service';
import { Observable } from "rxjs";

@Injectable({
    providedIn: 'root'
})
export class ServicesService{
    private urlEndpoint:string="http://localhost:8080/autospark/service"
    private httpHeaders = new HttpHeaders({'Content-Type':'application/json'})
    constructor(private http : HttpClient) {}

    getAll(): Observable<Services[]> {
        return this.http.get<Services[]>(this.urlEndpoint);
    }

    getById(id: number): Observable<Services> {
        return this.http.get<Services>(`${this.urlEndpoint}/${id}`, { headers: this.httpHeaders });
    }

    create(service: Services): Observable<Services> {
        return this.http.post<Services>(this.urlEndpoint, service, { headers: this.httpHeaders });
    }

    update(service: Services): Observable<Services> {
        return this.http.put<Services>(`${this.urlEndpoint}/${service.id}`, service, { headers: this.httpHeaders });
    }

    delete(id: number): Observable<void> {
        return this.http.delete<void>(`${this.urlEndpoint}/${id}`, { headers: this.httpHeaders });
    }
}