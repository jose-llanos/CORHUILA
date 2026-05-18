import { Routes } from '@angular/router';
import { HeaderComponent } from './header/header.component';
import { FooterComponent } from './footer/footer.component';
import { InventarioComponent } from './inventario/inventario.component';
import { FormComponentProducto } from './producto/productoform.component';
import { CarritoComponent } from './carrito/carrito.component';
import { ProductoComponent } from './producto/producto.component';

export const routes: Routes = [
    { path: '', redirectTo: '/producto', pathMatch: 'full' },  // Redirige a Inventario por defecto
 
    { path: 'header', component: HeaderComponent },  // Ruta existente de Header
    { path: 'footer', component: FooterComponent },  // Ruta existente de Footer

    // Nuevas rutas para Inventario, Producto y Carrito
    { path: 'inventario', component: InventarioComponent },  // Ruta para Inventario
    { path: 'producto', component: ProductoComponent },  // Ruta para mostrar un producto
    { path: 'producto/form', component: FormComponentProducto },
    { path: 'producto/form/:id', component: FormComponentProducto },
      // Ruta para editar Producto con ID
    { path: 'carrito', component: CarritoComponent },  // Ruta para el Carrito de compras
];


