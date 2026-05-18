import { Component, OnInit } from '@angular/core';
import { ServicesService } from './services.service';
import { Services } from './Service';
import { CommonModule } from '@angular/common';
import { Router, RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-services',
  imports: [CommonModule, RouterModule, FormsModule],
  templateUrl: './services.component.html',
  styleUrls: ['./services.component.css']
})
export class ServicesComponent implements OnInit {

  servicesList: Services[] = [];
  serviceForm: Services = {
    id: 0,
    name: '',
    description: '',
    price: 0,
    active: true,
    imageUrl: ''
  };

  userRole: string | null = null;

  constructor(private servicesService: ServicesService, private router: Router) {}

  ngOnInit(): void {
    if (typeof window !== 'undefined' && localStorage.getItem('userRole')) {
      this.userRole = localStorage.getItem('userRole')!;
    } else {
      this.userRole = null;
    }
    this.loadServices();
  }
  
  loadServices(): void {
    this.servicesService.getAll().subscribe(data => {
      this.servicesList = data;
    });
  }

  goToCreateService() {
    this.router.navigate(['/servicesform']);
  }

  updateService(service: Services): void {
    this.servicesService.update(service).subscribe(
      (updated) => {
        const index = this.servicesList.findIndex(s => s.id === updated.id);
        if (index !== -1) {
          this.servicesList[index] = updated;
        }
      },
      (error) => console.error('Error al actualizar el servicio:', error)
    );
  }

  deleteService(id: number): void {
    this.servicesService.delete(id).subscribe(
      () => {
        this.servicesList = this.servicesList.filter(s => s.id !== id);
      },
      (error) => console.error('Error al eliminar el servicio:', error)
    );
  }

  editService(service: Services): void {
    this.router.navigate(['/servicesform', service.id]);
  }

  toggleVisibility(service: Services): void {
    service.active = !service.active;
    this.updateService(service);
  }

  get visibleServices(): Services[] {
    if (this.userRole === 'ADMIN') {
      return this.servicesList;
    }
    return this.servicesList.filter(service => service.active);
  }
}