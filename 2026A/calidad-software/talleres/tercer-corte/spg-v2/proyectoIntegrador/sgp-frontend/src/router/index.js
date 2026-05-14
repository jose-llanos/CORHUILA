import { createRouter, createWebHistory } from 'vue-router'
import { useAuth } from '@/composables/useAuth'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/LoginView.vue'),
    meta: { public: true }
  },
  {
    path: '/',
    component: () => import('@/views/AppShell.vue'),
    meta: { requiresAuth: true },
    children: [
      { path: '', redirect: '/dashboard' },

      // ── Compartidas ───────────────────────────────
      {
        path: 'dashboard',
        name: 'Dashboard',
        component: () => import('@/views/DashboardView.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'perfil',
        name: 'Perfil',
        component: () => import('@/views/cliente/MiPerfilView.vue'),
        meta: { requiresAuth: true }
      },

      // ── Admin only ────────────────────────────────
      {
        path: 'usuarios',
        name: 'Usuarios',
        component: () => import('@/views/admin/UsuariosView.vue'),
        meta: { requiresAuth: true, requiresAdmin: true }
      },
      {
        path: 'equipos',
        name: 'Equipos',
        component: () => import('@/views/admin/EquiposView.vue'),
        meta: { requiresAuth: true, requiresAdmin: true }
      },
      {
        path: 'prestamos',
        name: 'Prestamos',
        component: () => import('@/views/admin/PrestamosView.vue'),
        meta: { requiresAuth: true, requiresAdmin: true }
      },
      {
        path: 'penalizaciones',
        name: 'Penalizaciones',
        component: () => import('@/views/admin/PenalizacionesView.vue'),
        meta: { requiresAuth: true, requiresAdmin: true }
      },

      // ── Cliente ───────────────────────────────────
      {
        path: 'catalogo',
        name: 'Catalogo',
        component: () => import('@/views/cliente/CatalogoView.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'mis-prestamos',
        name: 'MisPrestamos',
        component: () => import('@/views/cliente/MisPrestamosView.vue'),
        meta: { requiresAuth: true }
      },
      {
        path: 'mis-penalizaciones',
        name: 'MisPenalizaciones',
        component: () => import('@/views/cliente/MisPenalizacionesView.vue'),
        meta: { requiresAuth: true }
      },
    ]
  },
  {
    path: '/sin-acceso',
    name: 'SinAcceso',
    component: () => import('@/views/SinAccesoView.vue'),
    meta: { public: true }
  },
  { path: '/:pathMatch(.*)*', redirect: '/' }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to) => {
  const { isAuthenticated, isAdmin } = useAuth()
  if (to.meta.public) return true
  if (to.meta.requiresAuth && !isAuthenticated.value) {
    return { name: 'Login', query: { redirect: to.fullPath } }
  }
  if (to.meta.requiresAdmin && !isAdmin.value) {
    return { name: 'SinAcceso' }
  }
  return true
})

export default router
