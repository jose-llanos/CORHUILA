import { RenderMode, ServerRoute } from '@angular/ssr';

export const serverRoutes: ServerRoute[] = [
  {
    path: 'servicesform/:id',
    renderMode: RenderMode.Client  // Esta ruta NO se prerenderiza
  },
  {
    path: '**',
    renderMode: RenderMode.Prerender  // El resto SÍ se prerenderiza
  }
];
