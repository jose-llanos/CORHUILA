# SGP Lab — Frontend Vue 3 + Vuetify

Panel de administración para el Sistema de Gestión de Préstamos de laboratorio.

## Stack

- **Vue 3** (Composition API + `<script setup>`)
- **Vuetify 3** (Material Design, tema oscuro personalizado)
- **Vue Router 4** (con guardas de rol)
- **Axios** (capa de servicios hacia el backend Spring Boot)
- **Vite** (bundler)

## Estructura del proyecto

```
src/
├── composables/
│   └── useAuth.js          # Autenticación y manejo de roles
├── components/
│   ├── CrudDialog.vue      # Diálogo reutilizable para formularios
│   └── DeleteDialog.vue    # Confirmación de eliminación
├── plugins/
│   └── vuetify.js          # Configuración del tema SGP
├── router/
│   └── index.js            # Rutas con guardas por rol
├── services/
│   └── api.js              # Servicios HTTP para cada entidad
└── views/
    ├── LoginView.vue        # Login con selección de rol (demo)
    ├── AppShell.vue         # Layout principal con sidebar
    ├── DashboardView.vue    # Dashboard con estadísticas
    ├── SinAccesoView.vue    # Página 403
    └── admin/
        ├── UsuariosView.vue
        ├── EquiposView.vue
        ├── PrestamosView.vue
        └── PenalizacionesView.vue
```

## Instalación y uso

```bash
# Instalar dependencias (ya instaladas si descargaste el ZIP)
npm install

# Iniciar servidor de desarrollo (requiere backend en :8080)
npm run dev

# Build para producción
npm run build
```

## Configuración del backend

El proxy de Vite redirige `/api` → `http://localhost:8080`.
Si tu backend corre en otro puerto, edita `vite.config.js`:

```js
proxy: {
  '/api': {
    target: 'http://localhost:TU_PUERTO',
    changeOrigin: true
  }
}
```

## Autenticación (modo demo)

El login intenta buscar el usuario por email en `/api/usuarios`.
Si el backend no está disponible, crea una sesión local demo.
El **rol** se selecciona manualmente en el formulario de login.

En producción, reemplaza el método `handleLogin` en `LoginView.vue`
por una llamada real a tu endpoint `/api/auth/login`.

## Protección de rutas

| Ruta               | Requiere login | Requiere ADMINISTRADOR |
|--------------------|:--------------:|:---------------------:|
| `/dashboard`       | ✅             | ❌                    |
| `/usuarios`        | ✅             | ✅                    |
| `/equipos`         | ✅             | ✅                    |
| `/prestamos`       | ✅             | ✅                    |
| `/penalizaciones`  | ✅             | ✅                    |
