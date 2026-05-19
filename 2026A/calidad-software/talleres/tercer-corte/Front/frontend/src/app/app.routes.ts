import { Routes } from '@angular/router';
import { HeaderComponent } from './header/header.component';
import { FooterComponent } from './footer/footer.component';
import { HomeComponent } from './home/home.component';
import { LoginComponent } from './login_register/login.component';
import { ContactoComponent } from './contacto/contacto.component';
import { ServiciosComponent } from './servicios/servicios.component';
import { ReservaComponent } from './reserva/reserva.component';
import { FormComponent } from './login_register/form.componente';

import { AdminComponent } from './admin/admin.component';
import { RecoverpasswordComponent } from './recoverpassword/recoverpassword.component';
import { TaViewClienteComponent } from './ta-view-cliente/ta-view-cliente.component';
import { ServicioDetalleComponent } from './servicios/servicio-detalle/servicio-detalle.component';
import { TarifaComponent } from './servicios/tarifa/tarifa.component';
import { TarifaAdminComponent } from './tarifa-admin/tarifa-admin.component';
import { ViewClienteComponent } from './cliente/view-cliente.component';
import { ReservaConfirmadaComponent } from './reserva-confirmada/reserva-confirmada.component';
import { RegistroIngresoVehiculoComponent } from './registro-ingreso-vehiculo/registro-ingreso-vehiculo.component';
import { ListaIngresosComponent } from './lista-ingresos/lista-ingresos.component';




export const routes: Routes = [

    {path: '',redirectTo: '/home', pathMatch:'full'},
    {path: 'footer',component : FooterComponent},
    {path: 'home' ,component : HomeComponent},
    {path: 'header',component : HeaderComponent},
    {path: 'login',component : LoginComponent },
    {path: 'contacto', component : ContactoComponent},
    {path: 'servicios', component : ServiciosComponent},
    {path: 'reserva', component : ReservaComponent},
    {path: 'login/register', component: FormComponent },
    {path: 'admin', component : AdminComponent},
    {path: 'recuperarcontrasenia', component: RecoverpasswordComponent},
    {path: 'clienteview' , component : ViewClienteComponent},
    {path: 'servicio_detalle', component: ServicioDetalleComponent},
    {path:'servicio_tarifa', component: TarifaComponent},
    {path: 'tarifaAdmin', component: TarifaAdminComponent},
    {path: 'tarifaCliente', component: TaViewClienteComponent},
    {path: 'reservas', component: ReservaComponent},
    {path: 'reservas-confirmadas', component: ReservaConfirmadaComponent},
    {path: 'ingreso', component: RegistroIngresoVehiculoComponent},
    {path: 'lista-ingresos', component: ListaIngresosComponent}
   
];
