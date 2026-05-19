import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule, Validators} from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { ServicesService } from '../services.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-servicefrom',
  imports: [ReactiveFormsModule,FormsModule,CommonModule],
  templateUrl: './servicefrom.component.html',
  styleUrl: './servicefrom.component.css'
})
export class ServicefromComponent implements OnInit{

  serviceForm: FormGroup;
  isEdit: boolean = false;
  serviceId: number | null = null;

  constructor(
    private route: ActivatedRoute,
    private fb: FormBuilder,
    private servicesService: ServicesService,
    private router: Router
  ) {
    this.serviceForm = this.fb.group({
      name: ['', Validators.required],
      descripcion: ['', Validators.required],
      precioService: [0, [Validators.required, Validators.min(1)]],
      estadoService: [true, Validators.required],
      url_img: ['', [Validators.required, Validators.pattern('https?://.+')]]
    });
  }

  ngOnInit(): void {
    this.serviceId = Number(this.route.snapshot.paramMap.get('id'));

    if (this.serviceId) {
      this.isEdit = true;
      this.servicesService.getById(this.serviceId).subscribe(service => {
        this.serviceForm.patchValue(service);
      });
    }
  }

  guardarServicio() {
    if (this.serviceForm.invalid) {
      this.serviceForm.markAllAsTouched();
      return;
    }
  
    if (this.isEdit) {
      const updated = { ...this.serviceForm.value, id: this.serviceId };
      this.servicesService.update(updated).subscribe(() => {
        this.router.navigate(['/services']);
      });
    } else {
      this.servicesService.create(this.serviceForm.value).subscribe(() => {
        this.router.navigate(['/services']);  
      });
    }
  }
}
