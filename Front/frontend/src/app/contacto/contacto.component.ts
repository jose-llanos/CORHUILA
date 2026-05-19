import { Component } from '@angular/core';

@Component({
  selector: 'app-contacto',
  standalone: true,
  imports: [],
  templateUrl: './contacto.component.html',
  styleUrl: './contacto.component.css'
})
export class ContactoComponent {
  email: string = 'map_parking@gmail.com';
  socialHandle: string = '@map_parking';
  phone: string = '+573102303840';
}
