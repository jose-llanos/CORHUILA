<template>
  <div>
    <!-- Banner penalización -->
    <v-alert
      v-if="estaPenalizado"
      type="error"
      variant="tonal"
      class="mb-5"
      border="start"
      prominent
    >
      <template #title>
        <span class="mono">CUENTA PENALIZADA</span>
      </template>
      No puedes solicitar préstamos mientras tengas una penalización activa.
      Consulta la sección <strong>Mis Penalizaciones</strong> para más detalles.
    </v-alert>

    <!-- Toolbar -->
    <div class="d-flex align-center mb-5 gap-3 flex-wrap">
      <v-text-field
        v-model="search"
        prepend-inner-icon="mdi-magnify"
        placeholder="Buscar equipo..."
        variant="outlined"
        density="compact"
        bg-color="surface"
        hide-details
        style="max-width:280px"
      />
      <v-chip-group v-model="filtroEstado" mandatory>
        <v-chip value="TODOS" color="default" filter size="small" variant="tonal">Todos</v-chip>
        <v-chip value="DISPONIBLE" color="success" filter size="small" variant="tonal">Disponibles</v-chip>
        <v-chip value="EN_MANTENIMIENTO" color="warning" filter size="small" variant="tonal">En mantenimiento</v-chip>
      </v-chip-group>
    </div>

    <!-- Loading skeleton -->
    <v-row v-if="loading">
      <v-col v-for="n in 6" :key="n" cols="12" sm="6" lg="4">
        <v-skeleton-loader type="card" color="surface" />
      </v-col>
    </v-row>

    <!-- Cards de equipos -->
    <v-row v-else>
      <v-col
        v-for="equipo in filteredEquipos"
        :key="equipo.id"
        cols="12" sm="6" lg="4"
      >
        <v-card color="surface" class="equipo-card h-100" :class="{ 'equipo-card--disabled': equipo.cantidad === 0 }">
          <div class="equipo-card-accent" :class="`accent--${estadoColor(equipo.estado)}`" />

          <v-card-text class="pa-5">
            <div class="d-flex align-start justify-space-between mb-3">
              <div class="equipo-icon">
                <v-icon color="primary" size="22">mdi-desktop-classic</v-icon>
              </div>
              <v-chip :color="estadoColor(equipo.estado)" size="small" variant="tonal">
                {{ equipo.estado }}
              </v-chip>
            </div>

            <h3 class="text-subtitle-1 font-weight-semibold mb-1">{{ equipo.nombre }}</h3>
            <p class="mono text-caption text-medium-emphasis mb-4">{{ equipo.codigoInventario }}</p>

            <div class="stock-bar mb-4">
              <div class="d-flex justify-space-between mb-1">
                <span class="text-caption text-medium-emphasis">Stock disponible</span>
                <span class="mono text-caption" :class="equipo.cantidad > 0 ? 'text-success' : 'text-error'">
                  {{ equipo.cantidad }} unidades
                </span>
              </div>
              <v-progress-linear
                :model-value="Math.min(equipo.cantidad * 20, 100)"
                :color="equipo.cantidad > 3 ? 'success' : equipo.cantidad > 0 ? 'warning' : 'error'"
                rounded
                height="4"
                bg-color="surface-variant"
              />
            </div>

            <v-btn
              block
              :color="equipo.cantidad > 0 && !estaPenalizado ? 'primary' : 'default'"
              :disabled="equipo.cantidad === 0 || estaPenalizado"
              variant="flat"
              size="small"
              @click="abrirSolicitud(equipo)"
            >
              <v-icon start size="16">mdi-hand-extended-outline</v-icon>
              {{ equipo.cantidad === 0 ? 'Sin stock' : 'Solicitar préstamo' }}
            </v-btn>
          </v-card-text>
        </v-card>
      </v-col>

      <v-col v-if="filteredEquipos.length === 0" cols="12">
        <v-card color="surface" class="pa-8 text-center">
          <v-icon size="48" color="medium-emphasis" class="mb-3">mdi-magnify-close</v-icon>
          <p class="text-medium-emphasis">No se encontraron equipos</p>
        </v-card>
      </v-col>
    </v-row>

    <!-- Dialog solicitar préstamo -->
    <CrudDialog
      v-model="dialogSolicitud"
      title="Solicitar préstamo"
      icon="mdi-hand-extended-outline"
      color="primary"
      confirm-label="Confirmar solicitud"
      :loading="saving"
      @confirm="confirmarSolicitud"
    >
      <v-alert type="info" variant="tonal" density="compact" class="mb-4">
        Estás solicitando: <strong>{{ equipoSeleccionado?.nombre }}</strong>
      </v-alert>
      <v-row dense>
        <v-col cols="6">
          <v-text-field
            v-model="formSolicitud.fechaInicio"
            label="Fecha de inicio"
            type="date"
            prepend-inner-icon="mdi-calendar"
            bg-color="surface-variant"
          />
        </v-col>
        <v-col cols="6">
          <v-text-field
            v-model="formSolicitud.fechaFin"
            label="Fecha de devolución"
            type="date"
            prepend-inner-icon="mdi-calendar-end"
            bg-color="surface-variant"
          />
        </v-col>
      </v-row>
    </CrudDialog>

    <v-snackbar v-model="snack.show" :color="snack.color" location="bottom right" rounded="lg">
      {{ snack.text }}
    </v-snackbar>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, reactive } from 'vue'
import { equipoService, prestamoService, penalizacionService } from '@/services/api'
import { useAuth } from '@/composables/useAuth'
import CrudDialog from '@/components/CrudDialog.vue'

const { currentUser } = useAuth()

const equipos        = ref([])
const loading        = ref(false)
const saving         = ref(false)
const search         = ref('')
const filtroEstado   = ref('TODOS')
const estaPenalizado = ref(false)
const dialogSolicitud   = ref(false)
const equipoSeleccionado = ref(null)

const formSolicitud = reactive({
  fechaInicio: new Date().toISOString().split('T')[0],
  fechaFin: ''
})

const snack = reactive({ show: false, text: '', color: 'success' })

function showSnack(text, color = 'success') {
  snack.text = text; snack.color = color; snack.show = true
}

function estadoColor(e) {
  return { DISPONIBLE: 'success', PRESTADO: 'warning', EN_MANTENIMIENTO: 'error' }[e] || 'default'
}

const filteredEquipos = computed(() => {
  let list = equipos.value
  if (filtroEstado.value !== 'TODOS') list = list.filter(e => e.estado === filtroEstado.value)
  if (search.value) list = list.filter(e =>
    e.nombre.toLowerCase().includes(search.value.toLowerCase()) ||
    e.codigoInventario.toLowerCase().includes(search.value.toLowerCase())
  )
  return list
})

function abrirSolicitud(equipo) {
  equipoSeleccionado.value = equipo
  formSolicitud.fechaInicio = new Date().toISOString().split('T')[0]
  formSolicitud.fechaFin = ''
  dialogSolicitud.value = true
}

async function confirmarSolicitud() {
  saving.value = true
  try {
    await prestamoService.create({
      usuario: { id: currentUser.value.id },
      equipo:  { id: equipoSeleccionado.value.id },
      fechaInicio: formSolicitud.fechaInicio,
      fechaFin: formSolicitud.fechaFin || null,
      estado: 'ACTIVO'
    })
    showSnack('¡Préstamo solicitado exitosamente!')
    dialogSolicitud.value = false
    await loadData()
  } catch (e) {
    showSnack(e.response?.data?.message || 'Error al solicitar préstamo', 'error')
  } finally { saving.value = false }
}

async function loadData() {
  loading.value = true
  try {
    const [eq, pen] = await Promise.allSettled([
      equipoService.getAll(),
      currentUser.value?.id ? penalizacionService.checkUsuario(currentUser.value.id) : Promise.resolve({ data: false })
    ])
    if (eq.status === 'fulfilled') equipos.value = eq.value.data
    if (pen.status === 'fulfilled') estaPenalizado.value = pen.value.data
  } catch { showSnack('Error al cargar equipos', 'error') }
  finally { loading.value = false }
}

onMounted(loadData)
</script>

<style scoped>
.equipo-card {
  border: 1px solid rgba(255,255,255,0.06) !important;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
}
.equipo-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 32px rgba(0,212,255,0.08) !important;
}
.equipo-card--disabled { opacity: 0.6; }
.equipo-card-accent {
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
}
.accent--success { background: linear-gradient(90deg, transparent, #10b981, transparent); }
.accent--warning { background: linear-gradient(90deg, transparent, #f59e0b, transparent); }
.accent--error   { background: linear-gradient(90deg, transparent, #ef4444, transparent); }

.equipo-icon {
  width: 40px; height: 40px;
  background: rgba(0,212,255,0.08);
  border: 1px solid rgba(0,212,255,0.15);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
}
</style>
