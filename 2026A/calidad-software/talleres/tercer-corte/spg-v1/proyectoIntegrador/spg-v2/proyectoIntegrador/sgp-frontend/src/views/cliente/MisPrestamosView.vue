<template>
  <div>
    <!-- Resumen rápido -->
    <v-row class="mb-5">
      <v-col v-for="stat in stats" :key="stat.label" cols="6" sm="3">
        <v-card color="surface" class="stat-card pa-4">
          <div class="d-flex align-center gap-2 mb-2">
            <v-icon :color="stat.color" size="18">{{ stat.icon }}</v-icon>
            <span class="text-caption text-medium-emphasis">{{ stat.label }}</span>
          </div>
          <div class="mono text-h5 font-weight-bold">{{ stat.value }}</div>
        </v-card>
      </v-col>
    </v-row>

    <!-- Tabla de préstamos -->
    <v-card color="surface" class="data-card">
      <v-card-title class="pa-5 pb-3 d-flex align-center">
        <v-icon color="warning" class="mr-2">mdi-hand-extended-outline</v-icon>
        <span class="text-h6 font-weight-medium">Mis préstamos</span>
        <v-spacer />
        <v-chip-group v-model="filtro" mandatory>
          <v-chip v-for="f in filtros" :key="f.val" :value="f.val" :color="f.color" filter size="small" variant="tonal">
            {{ f.label }}
          </v-chip>
        </v-chip-group>
      </v-card-title>

      <v-data-table
        :headers="headers"
        :items="filteredPrestamos"
        :loading="loading"
        hover
        no-data-text="No tienes préstamos registrados"
      >
        <template #item.equipo="{ item }">
          <div class="d-flex align-center gap-2">
            <v-icon size="16" color="primary">mdi-desktop-classic</v-icon>
            <span>{{ item.equipo?.nombre || '—' }}</span>
          </div>
        </template>

        <template #item.estado="{ item }">
          <v-chip :color="estadoColor(item.estado)" size="small" variant="tonal" class="mono">
            <v-icon start size="10">mdi-circle</v-icon>
            {{ item.estado }}
          </v-chip>
        </template>

        <template #item.fechaFin="{ item }">
          <span :class="estaVencido(item) ? 'text-error font-weight-bold' : ''">
            {{ item.fechaFin || '—' }}
            <v-icon v-if="estaVencido(item)" size="14" color="error" class="ml-1">mdi-alert</v-icon>
          </span>
        </template>
      </v-data-table>
    </v-card>

    <v-snackbar v-model="snack.show" :color="snack.color" location="bottom right" rounded="lg">
      {{ snack.text }}
    </v-snackbar>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, reactive } from 'vue'
import { prestamoService } from '@/services/api'
import { useAuth } from '@/composables/useAuth'

const { currentUser } = useAuth()

const headers = [
  { title: 'ID',       key: 'id',          width: 60 },
  { title: 'Equipo',   key: 'equipo',      sortable: false },
  { title: 'Inicio',   key: 'fechaInicio' },
  { title: 'Devolución', key: 'fechaFin' },
  { title: 'Estado',   key: 'estado' },
]

const filtros = [
  { label: 'Todos',     val: null,       color: 'default' },
  { label: 'Activos',   val: 'ACTIVO',   color: 'warning' },
  { label: 'Devueltos', val: 'DEVUELTO', color: 'success' },
  { label: 'Vencidos',  val: 'VENCIDO',  color: 'error' },
]

const prestamos = ref([])
const loading   = ref(false)
const filtro    = ref(null)
const snack     = reactive({ show: false, text: '', color: 'success' })

const filteredPrestamos = computed(() =>
  filtro.value ? prestamos.value.filter(p => p.estado === filtro.value) : prestamos.value
)

const stats = computed(() => [
  { label: 'Total',     value: prestamos.value.length,                                          icon: 'mdi-history',          color: 'primary' },
  { label: 'Activos',   value: prestamos.value.filter(p => p.estado === 'ACTIVO').length,       icon: 'mdi-hand-extended',    color: 'warning' },
  { label: 'Devueltos', value: prestamos.value.filter(p => p.estado === 'DEVUELTO').length,     icon: 'mdi-check-circle',     color: 'success' },
  { label: 'Vencidos',  value: prestamos.value.filter(p => p.estado === 'VENCIDO').length,      icon: 'mdi-alert-circle',     color: 'error'   },
])

function estadoColor(e) {
  return { ACTIVO:'warning', DEVUELTO:'success', VENCIDO:'error', CANCELADO:'default' }[e] || 'default'
}

function estaVencido(item) {
  return item.estado === 'ACTIVO' && item.fechaFin && new Date(item.fechaFin) < new Date()
}

async function loadData() {
  loading.value = true
  try {
    const res = await prestamoService.getAll()
    // Filtra solo los del usuario actual
    prestamos.value = currentUser.value?.id === 0
      ? res.data
      : res.data.filter(p => p.usuario?.id === currentUser.value?.id)
  } catch {
    snack.text = 'Error al cargar préstamos'; snack.color = 'error'; snack.show = true
  } finally { loading.value = false }
}

onMounted(loadData)
</script>

<style scoped>
.stat-card { border: 1px solid rgba(255,255,255,0.06) !important; }
.data-card { border: 1px solid rgba(255,255,255,0.06) !important; }
</style>
