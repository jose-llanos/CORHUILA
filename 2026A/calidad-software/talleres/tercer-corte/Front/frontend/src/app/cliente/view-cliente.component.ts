import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { Router, RouterModule } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import { FormBuilder, FormGroup, FormsModule, Validators} from '@angular/forms';
import { MapServices } from '../admin/admin';
import { AdminService } from '../admin/admin.service';

@Component({
  selector: 'app-view-cliente',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './view-cliente.component.html',
  styleUrls: ['./view-cliente.component.css']
})
export class ViewClienteComponent implements OnInit {
  services: MapServices[] = []; // Lista de servicios
  serviceForm: FormGroup; // Formulario reactivo
  isEditing: boolean = false;
  editingServiceId: number | null = null;


  constructor(private fb: FormBuilder, private adminService: AdminService) {
    this.serviceForm = this.fb.group({
      name: ['', Validators.required],
      description: ['', Validators.required],
      image: ['', Validators.required]
    });
  }

  ngOnInit(): void {
    this.loadServices();
  }

  // Cargar servicios desde el backend
  loadServices(): void {
    this.adminService.getServices().subscribe((data) => {
      this.services = data;
    });
  }

  // Guardar o actualizar un servicio
  saveService(): void {
    // Verificar si el formulario es inválido
    if (this.serviceForm.invalid) {
      alert('Por favor, completa todos los campos obligatorios.');
      return;
    }
  
    // Preparar los datos para enviar al backend
    const serviceData: MapServices = {
      id: this.editingServiceId || 0, // Use 0 or another default value for new services
      name: this.serviceForm.value.name,
      description: this.serviceForm.value.description,
      image: this.serviceForm.value.image,
      price: 0 // Provide a default value or retrieve it from the form if applicable
    };
  
    // Determinar si es una creación o una actualización
    const request = this.isEditing && this.editingServiceId
      ? this.adminService.updateService(this.editingServiceId, serviceData)
      : this.adminService.createService(serviceData);
  
    // Enviar la solicitud al backend
    request.subscribe({
      next: (newService) => {
        if (this.isEditing) {
          // Actualizar el servicio en la lista
          const index = this.services.findIndex(s => s.id === this.editingServiceId);
          if (index !== -1) {
            this.services[index] = newService;
          }
        } else {
          // Agregar el nuevo servicio a la lista
          this.services.push(newService);
        }
  
        // Resetear el formulario y estado de edición
        this.resetForm();
        alert('El servicio se guardó correctamente.');
      },
      error: (err) => {
        console.error('Error al guardar el servicio:', err);
        alert('Ocurrió un error al guardar el servicio. Por favor, inténtalo de nuevo.');
      }
    });
  }

  // Editar un servicio
  editService(service: any): void {
    this.isEditing = true;
    this.editingServiceId = service.id;
    this.serviceForm.patchValue(service);
  }

  // Eliminar un servicio
  deleteService(id: number): void {
    this.adminService.deleteService(id).subscribe(() => {
      this.loadServices();
    });
  }

  // Resetear el formulario
  resetForm(): void {
    this.isEditing = false;
    this.editingServiceId = null;
    this.serviceForm.reset();
  }

  // Cerrar sesión
  logout(): void {
    console.log('Sesión cerrada');
  }
}
