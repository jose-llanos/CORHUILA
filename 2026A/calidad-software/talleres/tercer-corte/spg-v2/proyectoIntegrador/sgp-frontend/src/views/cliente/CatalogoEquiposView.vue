<template>
  <div>
    <!-- Header con búsqueda -->
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
        <v-chip value="PRESTADO" color="warning" filter size="small" variant="tonal">Prestados</v-chip>
        <v-chip value="EN_MANTENIMIENTO" color="error" filter size="small" variant="tonal">Mantenimiento</v-chip>
      </v-chip-group>
      <v-spacer />
      <v-btn-toggle v-model="viewMode" mandatory density="compact" variant="outlined">
        <v-btn icon="mdi-view-grid-outline" value="grid" />
        <v-btn icon="mdi-format-list-bulleted" value="list" />
      </v-btn-toggle>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="d-flex justify-center py-12">
      <v-progress-circular indeterminate color="primary" size="48" />
    </div>

    <!-- Grid view -->
    <v-row v-else-if="viewMode === 'grid'">
      <v-col
        v-for="equipo in filteredEquipos"
        :key="equipo.id"
        cols="12" sm="6" md="4" lg="3"
      >
        <v-card color="surface" class="equipo-card h-100" :class="`equipo-card--${equipo.estado}`">
          <div class="equipo-card-header">
            <div class="equipo-icon">
              <v-icon size="32" :color="estadoColor(equipo.estado)">mdi-desktop-classic</v-icon>
            </div>
            <v-chip :color="estadoColor(equipo.estado)" size="x-small" variant="tonal" class="mono">
              {{ equipo.estado }}
            </v-chip>
          </div>

          <v-card-text class="pt-2">
            <h3 class="text-body-1 font-weight-semibold mb-1">{{ equipo.nombre }}</h3>
            <p class="text-caption mono text-primary mb-3">{{ equipo.codigoInventario }}</p>

            <div class="d-flex align-center justify-space-between">
              <div>
                <div class="text-caption text-medium-emphasis">Stock disponible</div>
                <div class="text-h6 mono font-weight-bold" :class="equipo.cantidad > 0 ? 'text-success' : 'text-error'">
                  {{ equipo.cantidad }}
                </div>
              </div>
              <v-btn
                v-if="equipo.estado === 'DISPONIBLE' && equipo.cantidad > 0"
                color="primary"
                size="small"
                variant="flat"
                @click="solicitarPrestamo(equipo)"
              >
                Solicitar
              </v-btn>
              <v-btn v-else size="small" variant="tonal" color="default" disabled>
                No disponible
              </v-btn>
            </div>
          </v-card-text>
        </v-card>
      </v-col>

      <v-col v-if="filteredEquipos.length === 0" cols="12" class="text-center py-12">
        <v-icon size="64" color="medium-emphasis" class="mb-3">mdi-magnify-close</v-icon>
        <p class="text-medium-emphasis">No se encontraron equipos</p>
      </v-col>
    </v-row>

    <!-- List view -->
    <v-card v-else color="surface" style="border:1px solid rgba(255,255,255,0.06)">
      <v-data-table
        :headers="headers"
        :items="filteredEquipos"
        :search="search"
        hover
        no-data-text="No hay equipos"
      >
        <template #item.estado="{ item }">
          <v-chip :color="estadoColor(item.estado)" size="small" variant="tonal">
            {{ item.estado }}
          </v-chip>
        </template>
        <template #item.cantidad="{ item }">
          <span class="mono font-weight-bold" :class="item.cantidad > 0 ? 'text-success' : 'text-error'">
            {{ item.cantidad }}
          </span>
        </template>
        <template #item.codigoInventario="{ item }">
          <span class="mono text-primary text-caption">{{ item.codigoInventario }}</span>
        </template>
        <template #item.actions="{ item }">
          <v-btn
            v-if="item.estado === 'DISPONIBLE' && item.cantidad > 0"
            color="primary" size="small" variant="flat"
            @click="solicitarPrestamo(item)"
          >
            Solicitar préstamo
          </v-btn>
          <v-chip v-else size="small" variant="tonal" color="default">No disponible</v-chip>
        </template>
      </v-data-table>
    </v-card>

    <!-- Dialog solicitar préstamo -->
    <CrudDialog
      v-model="dialogPrestamo"
      title="Solicitar préstamo"
      icon="mdi-hand-extended-outline"
      color="primary"
      confirm-label="Confirmar solicitud"
      :loading="saving"
      @confirm="confirmarPrestamo"
    >
      <v-alert type="info" variant="tonal" density="compact" class="mb-4 text-body-2">
        Estás solicitando el equipo: <strong>{{ equipoSeleccionado?.nombre }}</strong>
      </v-alert>

      <v-alert v-if="tienePrestamoActivo" type="warning" variant="tonal" density="compact" class="mb-4 text-body-2">
        Ya tienes un préstamo activo. Debes devolverlo antes de solicitar otro.
      </v-alert>

      <v-alert v-if="estasPenalizado" type="error" variant="tonal" density="compact" class="mb-4 text-body-2">
        Tienes una penalización activa. No puedes solicitar préstamos.
      </v-alert>

      <v-row dense v-if="!tienePrestamoActivo && !estasPenalizado">
        <v-col cols="6">
          <v-text-field
            v-model="formPrestamo.fechaInicio"
            label="Fecha inicio"
            type="date"
            prepend-inner-icon="mdi-calendar"
            bg-color="surface-variant"
          />
        </v-col>
        <v-col cols="6">
          <v-text-field
            v-model="formPrestamo.fechaFin"
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
import { ref, reactive, computed, onMounted } from 'vue'
import { equipoService, prestamoService, penalizacionService } from '@/services/api'
import { useAuth } from '@/composables/useAuth'
import CrudDialog from '@/components/CrudDialog.vue'

const { currentUser } = useAuth()

const equipos        = ref([])
const loading        = ref(false)
const saving         = ref(false)
const search         = ref('')
const filtroEstado   = ref('TODOS')
const viewMode       = ref('grid')
const dialogPrestamo = ref(false)
const equipoSeleccionado = ref(null)
const tienePrestamoActivo = ref(false)
const estasPenalizado     = ref(false)

const formPrestamo = reactive({
  fechaInicio: new Date().toISOString().split('T')[0],
  fechaFin: ''
})
const snack = reactive({ show: false, text: '', color: 'success' })

const headers = [
  { title: 'Código',   key: 'codigoInventario' },
  { title: 'Nombre',   key: 'nombre', sortable: true },
  { title: 'Stock',    key: 'cantidad', sortable: true },
  { title: 'Estado',   key: 'estado' },
  { title: '',         key: 'actions', sortable: false, align: 'end' },
]

const filteredEquipos = computed(() => {
  let list = equipos.value
  if (filtroEstado.value !== 'TODOS') list = list.filter(e => e.estado === filtroEstado.value)
  if (search.value) list = list.filter(e =>
    e.nombre.toLowerCase().includes(search.value.toLowerCase()) ||
    e.codigoInventario.toLowerCase().includes(search.value.toLowerCase())
  )
  return list
})

function estadoColor(e) {
  return { DISPONIBLE: 'success', PRESTADO: 'warning', EN_MANTENIMIENTO: 'error' }[e] || 'default'
}
function showSnack(text, color = 'success') {
  snack.text = text; snack.color = color; snack.show = true
}

async function solicitarPrestamo(equipo) {
  equipoSeleccionado.value = equipo
  formPrestamo.fechaInicio = new Date().toISOString().split('T')[0]
  formPrestamo.fechaFin = ''

  try {
    const [pen, pres] = await Promise.allSettled([
      penalizacionService.checkUsuario(currentUser.value?.id),
      prestamoService.getAll()
    ])
    estasPenalizado.value     = pen.status === 'fulfilled' ? pen.value.data : false
    tienePrestamoActivo.value = pres.status === 'fulfilled'
      ? pres.value.data.some(p => p.usuario?.id === currentUser.value?.id && p.estado === 'ACTIVO')
      : false
  } catch { estasPenalizado.value = false; tienePrestamoActivo.value = false }

  dialogPrestamo.value = true
}

async function confirmarPrestamo() {
  if (tienePrestamoActivo.value || estasPenalizado.value) {
    dialogPrestamo.value = false; return
  }
  saving.value = true
  try {
    await prestamoService.create({
      usuario: { id: currentUser.value?.id },
      equipo:  { id: equipoSeleccionado.value?.id },
      fechaInicio: formPrestamo.fechaInicio,
      fechaFin: formPrestamo.fechaFin || null,
      estado: 'ACTIVO'
    })
    showSnack('¡Préstamo solicitado exitosamente!')
    dialogPrestamo.value = false
    await loadData()
  } catch (e) {
    showSnack(e.response?.data?.message || 'Error al solicitar préstamo', 'error')
  } finally { saving.value = false }
}

async function loadData() {
  loading.value = true
  try {
    const res = await equipoService.getAll()
    equipos.value = res.data
  } catch { showSnack('Error al cargar equipos', 'error') }
  finally { loading.value = false }
}

onMounted(loadData)
</script>

<style scoped>
.equipo-card {
  border: 1px solid rgba(255,255,255,0.06) !important;
  transition: transform 0.2s, box-shadow 0.2s;
}
.equipo-card:hover { transform: translateY(-3px); }
.equipo-card--DISPONIBLE:hover { box-shadow: 0 8px 24px rgba(16,185,129,0.1) !important; }
.equipo-card--PRESTADO:hover   { box-shadow: 0 8px 24px rgba(245,158,11,0.1) !important; }

.equipo-card-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 16px 16px 8px;
}
.equipo-icon {
  width: 52px; height: 52px;
  background: rgba(255,255,255,0.05);
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
}
</style>
