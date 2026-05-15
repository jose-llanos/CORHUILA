import { Routes } from '@angular/router';
import { HeaderComponent } from './components/header/header.component';
import { FooterComponent } from './components/footer/footer.component';
import { ServicesComponent } from './components/services/services.component';
import { RegisterComponent } from './components/register/register.component';
import { LoginComponent } from './components/login/login.component';
import { RecuperarcontraseniaComponent } from './components/recuperarcontrasenia/recuperarcontrasenia.component';
import { ServicefromComponent } from './components/services/servicefrom/servicefrom.component';
import { ReserveComponent } from './components/reserve/reserve.component';
import { TablaReservasComponent } from './components/tabla-reservas/tabla-reservas.component';
import { UsersTableComponent } from './components/users-table/users-table.component';

export const routes: Routes = [
    {path:'', redirectTo:'/services', pathMatch:'full'},
    {path:'header',component:HeaderComponent},
    {path:'footer',component:FooterComponent},
    {path:'services',component:ServicesComponent},
    {path:'servicesform',component:ServicefromComponent},
    {path:'servicesform/:id',component:ServicefromComponent},
    {path:'register',component:RegisterComponent},
    {path:'login',component:LoginComponent},
    {path:'recuperarcontrasenia', component: RecuperarcontraseniaComponent},
    {path:'reserves',component:ReserveComponent},
    {path:'tablareservas',component:TablaReservasComponent},
    {path: 'users-table', component: UsersTableComponent}
];