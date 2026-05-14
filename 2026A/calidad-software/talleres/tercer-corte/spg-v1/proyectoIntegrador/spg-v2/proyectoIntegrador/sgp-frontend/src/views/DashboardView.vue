<template>
  <div>
    <!-- ══ VISTA ADMINISTRADOR ══════════════════════════════ -->
    <template v-if="isAdmin">
      <v-row class="mb-6">
        <v-col v-for="stat in adminStats" :key="stat.label" cols="12" sm="6" lg="3">
          <v-card color="surface" class="stat-card pa-5" :class="`stat-card--${stat.color}`">
            <div class="d-flex align-center justify-space-between mb-4">
              <div class="stat-icon" :style="`background:rgba(var(--v-theme-${stat.color}),0.12)`">
                <v-icon :color="stat.color" size="20">{{ stat.icon }}</v-icon>
              </div>
              <v-chip :color="stat.color" variant="tonal" size="x-small" class="mono">
                {{ stat.trend }}
              </v-chip>
            </div>
            <div class="stat-value mono">{{ stat.loading ? '—' : stat.value }}</div>
            <div class="stat-label">{{ stat.label }}</div>
          </v-card>
        </v-col>
      </v-row>

      <v-card color="surface" class="pa-6">
        <div class="d-flex align-center mb-5">
          <v-icon color="primary" class="mr-2">mdi-lightning-bolt</v-icon>
          <h3 class="text-h6 font-weight-medium">Acceso rápido</h3>
        </div>
        <v-row>
          <v-col v-for="action in adminActions" :key="action.to" cols="6" sm="3">
            <v-btn :to="action.to" :color="action.color" variant="tonal" block size="large" class="quick-btn">
              <div class="d-flex flex-column align-center py-2">
                <v-icon :size="28" class="mb-2">{{ action.icon }}</v-icon>
                <span class="text-caption">{{ action.label }}</span>
              </div>
            </v-btn>
          </v-col>
        </v-row>
      </v-card>
    </template>

    <!-- ══ VISTA CLIENTE ═══════════════════════════════════ -->
    <template v-else>
      <!-- Banner penalización -->
      <v-alert
        v-if="estaPenalizado"
        type="error"
        variant="tonal"
        border="start"
        prominent
        class="mb-5"
      >
        <template #title><span class="mono">CUENTA RESTRINGIDA</span></template>
        Tienes una penalización activa. No puedes solicitar nuevos préstamos.
        <v-btn variant="text" size="small" to="/mis-penalizaciones" class="ml-2">Ver detalles</v-btn>
      </v-alert>

      <!-- Saludo -->
      <v-card color="surface" class="welcome-card pa-6 mb-5">
        <div class="welcome-glow" />
        <div class="d-flex align-center gap-4">
          <v-avatar size="56" color="primary">
            <span class="text-h5 font-weight-bold text-black">{{ userInitial }}</span>
          </v-avatar>
          <div>
            <h2 class="text-h6 font-weight-semibold">Bienvenido, {{ currentUser?.nombre }}</h2>
            <p class="text-body-2 text-medium-emphasis">
              {{ estaPenalizado ? 'Tu cuenta tiene restricciones activas.' : 'Tu cuenta está activa. ¡Puedes solicitar equipos!' }}
            </p>
          </div>
        </div>
      </v-card>

      <!-- Stats cliente -->
      <v-row class="mb-5">
        <v-col v-for="stat in clienteStats" :key="stat.label" cols="6" sm="3">
          <v-card color="surface" class="stat-card pa-4">
            <div class="d-flex align-center gap-2 mb-2">
              <v-icon :color="stat.color" size="18">{{ stat.icon }}</v-icon>
              <span class="text-caption text-medium-emphasis">{{ stat.label }}</span>
            </div>
            <div class="mono text-h5 font-weight-bold">{{ stat.loading ? '—' : stat.value }}</div>
          </v-card>
        </v-col>
      </v-row>

      <!-- Acciones cliente -->
      <v-row>
        <v-col v-for="action in clienteActions" :key="action.to" cols="12" sm="4">
          <v-card
            :to="action.to"
            color="surface"
            class="action-card pa-5"
            :class="`action-card--${action.color}`"
            hover
          >
            <v-icon :color="action.color" size="32" class="mb-3">{{ action.icon }}</v-icon>
            <h3 class="text-subtitle-1 font-weight-semibold mb-1">{{ action.label }}</h3>
            <p class="text-caption text-medium-emphasis">{{ action.desc }}</p>
          </v-card>
        </v-col>
      </v-row>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useAuth } from '@/composables/useAuth'
import { usuarioService, equipoService, prestamoService, penalizacionService } from '@/services/api'

const { currentUser, isAdmin } = useAuth()

const userInitial    = computed(() => currentUser.value?.nombre?.charAt(0).toUpperCase() || 'U')
const estaPenalizado = ref(false)

// ── Admin stats ──────────────────────────────────────────
const adminStats = ref([
  { label: 'Usuarios registrados',  icon: 'mdi-account-group',    color: 'primary',   value: 0, trend: 'TOTAL',  loading: true },
  { label: 'Equipos en inventario', icon: 'mdi-desktop-classic',  color: 'secondary',  value: 0, trend: 'STOCK',  loading: true },
  { label: 'Préstamos activos',     icon: 'mdi-hand-extended',    color: 'warning',   value: 0, trend: 'ACTIVO', loading: true },
  { label: 'Penalizaciones activas',icon: 'mdi-alert-circle',     color: 'error',     value: 0, trend: 'ACTIVA', loading: true },
])

const adminActions = [
  { to: '/usuarios',       icon: 'mdi-account-plus-outline', label: 'Gestionar Usuarios',  color: 'primary' },
  { to: '/equipos',        icon: 'mdi-plus-box-outline',      label: 'Gestionar Equipos',   color: 'secondary' },
  { to: '/prestamos',      icon: 'mdi-hand-extended-outline', label: 'Ver Préstamos',       color: 'warning' },
  { to: '/penalizaciones', icon: 'mdi-shield-alert-outline',  label: 'Penalizaciones',      color: 'error' },
]

// ── Cliente stats ────────────────────────────────────────
const clienteStats = ref([
  { label: 'Mis préstamos',  icon: 'mdi-history',       color: 'primary',  value: 0, loading: true },
  { label: 'Activos',        icon: 'mdi-hand-extended', color: 'warning',  value: 0, loading: true },
  { label: 'Devueltos',      icon: 'mdi-check-circle',  color: 'success',  value: 0, loading: true },
  { label: 'Penalizaciones', icon: 'mdi-alert-circle',  color: 'error',    value: 0, loading: true },
])

const clienteActions = [
  { to: '/catalogo',           icon: 'mdi-view-grid-outline',      label: 'Catálogo de equipos',   desc: 'Explora los equipos disponibles y solicita un préstamo',  color: 'primary' },
  { to: '/mis-prestamos',      icon: 'mdi-hand-extended-outline',  label: 'Mis préstamos',         desc: 'Consulta el historial y estado de tus préstamos',          color: 'warning' },
  { to: '/mis-penalizaciones', icon: 'mdi-shield-alert-outline',   label: 'Mis penalizaciones',    desc: 'Revisa si tienes restricciones activas en tu cuenta',      color: 'error' },
]

onMounted(async () => {
  if (isAdmin.value) {
    const [usuarios, equipos, prestamos, penalizaciones] = await Promise.allSettled([
      usuarioService.getAll(),
      equipoService.getAll(),
      prestamoService.getAll(),
      penalizacionService.getAll(),
    ])
    if (usuarios.status === 'fulfilled')      { adminStats.value[0].value = usuarios.value.data.length;                                                  adminStats.value[0].loading = false }
    if (equipos.status === 'fulfilled')       { adminStats.value[1].value = equipos.value.data.length;                                                   adminStats.value[1].loading = false }
    if (prestamos.status === 'fulfilled')     { adminStats.value[2].value = prestamos.value.data.filter(p => p.estado === 'ACTIVO').length;              adminStats.value[2].loading = false }
    if (penalizaciones.status === 'fulfilled'){ adminStats.value[3].value = penalizaciones.value.data.filter(p => p.estado === 'ACTIVA').length;         adminStats.value[3].loading = false }
  } else {
    // Cliente: carga sus propios datos
    const uid = currentUser.value?.id
    const [prestamos, penalizaciones, pen_activa] = await Promise.allSettled([
      prestamoService.getAll(),
      penalizacionService.getAll(),
      uid ? penalizacionService.checkUsuario(uid) : Promise.resolve({ data: false })
    ])
    if (prestamos.status === 'fulfilled') {
      const misPrestamos = uid === 0 ? prestamos.value.data : prestamos.value.data.filter(p => p.usuario?.id === uid)
      clienteStats.value[0].value = misPrestamos.length
      clienteStats.value[1].value = misPrestamos.filter(p => p.estado === 'ACTIVO').length
      clienteStats.value[2].value = misPrestamos.filter(p => p.estado === 'DEVUELTO').length
      clienteStats.value.forEach((s, i) => { if (i < 3) s.loading = false })
    }
    if (penalizaciones.status === 'fulfilled') {
      const misPen = uid === 0 ? penalizaciones.value.data : penalizaciones.value.data.filter(p => p.usuario?.id === uid)
      clienteStats.value[3].value = misPen.filter(p => p.estado === 'ACTIVA').length
      clienteStats.value[3].loading = false
    }
    if (pen_activa.status === 'fulfilled') estaPenalizado.value = pen_activa.value.data
  }
})
</script>

<style scoped>
.stat-card {
  border: 1px solid rgba(255,255,255,0.06) !important;
  transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover { transform: translateY(-2px); }
.stat-card--primary:hover   { box-shadow: 0 8px 32px rgba(0,212,255,0.1)  !important; }
.stat-card--secondary:hover { box-shadow: 0 8px 32px rgba(255,107,53,0.1) !important; }
.stat-card--warning:hover   { box-shadow: 0 8px 32px rgba(245,158,11,0.1) !important; }
.stat-card--error:hover     { box-shadow: 0 8px 32px rgba(239,68,68,0.1)  !important; }

.stat-icon {
  width: 40px; height: 40px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
}
.stat-value { font-size: 2rem; font-weight: 700; color: #e2e8f0; line-height: 1; margin-bottom: 4px; }
.stat-label { font-size: 0.8rem; color: rgba(226,232,240,0.5); }

.quick-btn { height: auto !important; padding: 12px !important; }

.welcome-card {
  border: 1px solid rgba(0,212,255,0.15) !important;
  position: relative; overflow: hidden;
}
.welcome-glow {
  position: absolute; top: -60px; right: -60px;
  width: 200px; height: 200px;
  background: radial-gradient(circle, rgba(0,212,255,0.08), transparent 70%);
  pointer-events: none;
}

.action-card {
  border: 1px solid rgba(255,255,255,0.06) !important;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
  text-decoration: none;
}
.action-card:hover { transform: translateY(-3px); }
.action-card--primary:hover  { box-shadow: 0 8px 32px rgba(0,212,255,0.12)  !important; border-color: rgba(0,212,255,0.2) !important; }
.action-card--warning:hover  { box-shadow: 0 8px 32px rgba(245,158,11,0.12) !important; border-color: rgba(245,158,11,0.2) !important; }
.action-card--error:hover    { box-shadow: 0 8px 32px rgba(239,68,68,0.12)  !important; border-color: rgba(239,68,68,0.2)  !important; }
</style>
