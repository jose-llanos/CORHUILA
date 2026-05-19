import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule, AbstractControl } from '@angular/forms';
import { Router } from '@angular/router';
import Swal from 'sweetalert2';
import { CommonModule } from '@angular/common';
import { User } from './user';
import { HttpClientModule } from '@angular/common/http';
import { UserService } from './user.service';

@Component({
  selector: 'app-form',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, HttpClientModule], 
  templateUrl: './form.component.html',
  styleUrls: ['./form.component.css']
})
export class FormComponent implements OnInit {
  public registroForm!: FormGroup;
  public showPassword: boolean = false;
  

  constructor(
    private fb: FormBuilder, 
    private router: Router,
    private userService: UserService // Inyecta el servicio
  ) {}

  ngOnInit() {
    this.registroForm = this.fb.group({
      name: ['', Validators.required],
      lastname: ['', Validators.required],
      phone: ['', [Validators.required, Validators.pattern('^[0-9]+$')]],
      plate: ['', [Validators.required, Validators.pattern('^\\S{1,8}$')]],
      typecar: ['', Validators.required], // Se agregó la validación
      email: ['', [Validators.required, Validators.email]],
      rol: ['', Validators.required], // Rol predefinido
      password: ['', [Validators.required, Validators.minLength(6)]],
      confirmarPassword: ['', Validators.required]
    }, { validators: this.passwordsMatchValidator });
  }

  get formControls() {
    return this.registroForm.controls;
  }

  private passwordsMatchValidator(form: FormGroup) {
    const password = form.get('password')?.value;
    const confirmPassword = form.get('confirmarPassword')?.value;
    return password === confirmPassword ? null : { noCoincide: true };
  }

  public create(): void {
    if (this.registroForm.invalid) {
      Swal.fire({
        title: 'Error',
        text: 'Por favor, complete todos los campos obligatorios correctamente.',
        icon: 'error'
      });
      return;
    }

    const user: User = this.registroForm.value;

    this.userService.create(user).subscribe({
      next: () => {
        Swal.fire({
          title: 'Registro exitoso',
          text: 'Usuario registrado correctamente',
          icon: 'success',
          confirmButtonText: 'Volver a inicio'
        }).then(() => {
          this.router.navigate(['home']);
        });
      },
      error: (err) => {
        console.error('Error en la creación del usuario:', err);
        Swal.fire({
          title: 'Error',
          text: 'Hubo un problema al registrar el usuario.',
          icon: 'error'
        });
      }
    });
  }

  togglePasswordVisibility() {
    this.showPassword = !this.showPassword;

   
  }

  navigateToLogin() {
    this.router.navigate(['/login']);
  }
}