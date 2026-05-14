<template>
  <div>
    <div class="d-flex align-center mb-5 gap-3 flex-wrap">
      <v-text-field
        v-model="search"
        prepend-inner-icon="mdi-magnify"
        placeholder="Buscar préstamo..."
        variant="outlined"
        density="compact"
        bg-color="surface"
        hide-details
        style="max-width:280px"
      />
      <v-chip-group v-model="filtroEstado" mandatory>
        <v-chip v-for="f in filtros" :key="f.val" :value="f.val" :color="f.color" filter size="small" variant="tonal">
          {{ f.label }}
        </v-chip>
      </v-chip-group>
      <v-spacer />
      <v-btn color="warning" prepend-icon="mdi-hand-extended" @click="openCreate">
        Nuevo préstamo
      </v-btn>
    </div>

    <v-card color="surface" class="data-card">
      <v-data-table
        :headers="headers"
        :items="filteredPrestamos"
        :search="search"
        :loading="loading"
        hover
        no-data-text="No hay préstamos registrados"
      >
        <template #item.estado="{ item }">
          <v-chip :color="estadoColor(item.estado)" size="small" variant="tonal" class="mono">
            {{ item.estado }}
          </v-chip>
        </template>

        <template #item.equipo="{ item }">
          <span class="text-medium-emphasis">{{ item.equipo?.nombre || `ID: ${item.equipo?.id}` }}</span>
        </template>

        <template #item.usuario="{ item }">
          <div class="d-flex align-center">
            <v-avatar size="22" color="primary" class="mr-2">
              <span style="font-size:10px;color:black;font-weight:700">{{ item.usuario?.nombre?.charAt(0) || 'U' }}</span>
            </v-avatar>
            <span>{{ item.usuario?.nombre || `ID: ${item.usuario?.id}` }}</span>
          </div>
        </template>

        <template #item.actions="{ item }">
          <div class="d-flex gap-1">
            <v-btn icon="mdi-pencil-outline" variant="text" size="small" color="warning" @click="openEdit(item)" />
            <v-btn icon="mdi-trash-can-outline" variant="text" size="small" color="error" @click="openDelete(item)" />
          </div>
        </template>
      </v-data-table>
    </v-card>

    <!-- Create/Edit Dialog -->
    <CrudDialog
      v-model="dialog"
      :title="isEdit ? 'Editar préstamo' : 'Nuevo préstamo'"
      icon="mdi-hand-extended-outline"
      color="warning"
      :loading="saving"
      @confirm="handleSave"
    >
      <v-row dense>
        <v-col cols="12">
          <v-select
            v-model="form.usuarioId"
            label="Usuario"
            :items="usuarios"
            item-title="nombre"
            item-value="id"
            prepend-inner-icon="mdi-account"
            bg-color="surface-variant"
          />
        </v-col>
        <v-col cols="12">
          <v-select
            v-model="form.equipoId"
            label="Equipo"
            :items="equipos"
            item-title="nombre"
            item-value="id"
            prepend-inner-icon="mdi-desktop-classic"
            bg-color="surface-variant"
          />
        </v-col>
        <v-col cols="6">
          <v-text-field v-model="form.fechaInicio" label="Fecha inicio" type="date" prepend-inner-icon="mdi-calendar" bg-color="surface-variant" />
        </v-col>
        <v-col cols="6">
          <v-text-field v-model="form.fechaFin" label="Fecha fin" type="date" prepend-inner-icon="mdi-calendar-end" bg-color="surface-variant" />
        </v-col>
        <v-col cols="12">
          <v-select
            v-model="form.estado"
            label="Estado"
            :items="['ACTIVO','DEVUELTO','VENCIDO','CANCELADO']"
            prepend-inner-icon="mdi-toggle-switch"
            bg-color="surface-variant"
          />
        </v-col>
      </v-row>
    </CrudDialog>

    <DeleteDialog
      v-model="deleteDialog"
      :item-name="`Préstamo #${selected?.id}`"
      :loading="saving"
      @confirm="handleDelete"
    />

    <v-snackbar v-model="snack.show" :color="snack.color" location="bottom right" rounded="lg">
      {{ snack.text }}
    </v-snackbar>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { prestamoService, usuarioService, equipoService } from '@/services/api'
import CrudDialog from '@/components/CrudDialog.vue'
import DeleteDialog from '@/components/DeleteDialog.vue'

const headers = [
  { title: 'ID',          key: 'id',         width: 60 },
  { title: 'Usuario',     key: 'usuario',    sortable: false },
  { title: 'Equipo',      key: 'equipo',     sortable: false },
  { title: 'Fecha inicio',key: 'fechaInicio' },
  { title: 'Fecha fin',   key: 'fechaFin' },
  { title: 'Estado',      key: 'estado' },
  { title: 'Acciones',    key: 'actions', sortable: false, align: 'end' },
]

const filtros = [
  { label: 'Todos',     val: null,       color: 'default' },
  { label: 'Activos',   val: 'ACTIVO',   color: 'warning' },
  { label: 'Devueltos', val: 'DEVUELTO', color: 'success' },
  { label: 'Vencidos',  val: 'VENCIDO',  color: 'error' },
]

const prestamos  = ref([])
const usuarios   = ref([])
const equipos    = ref([])
const loading    = ref(false)
const saving     = ref(false)
const search     = ref('')
const dialog     = ref(false)
const deleteDialog = ref(false)
const isEdit     = ref(false)
const selected   = ref(null)
const filtroEstado = ref(null)

const form = reactive({ usuarioId: null, equipoId: null, fechaInicio: '', fechaFin: '', estado: 'ACTIVO' })
const snack = reactive({ show: false, text: '', color: 'success' })

const filteredPrestamos = computed(() =>
  filtroEstado.value ? prestamos.value.filter(p => p.estado === filtroEstado.value) : prestamos.value
)

function estadoColor(e) {
  return { ACTIVO:'warning', DEVUELTO:'success', VENCIDO:'error', CANCELADO:'default' }[e] || 'default'
}
function showSnack(text, color = 'success') {
  snack.text = text; snack.color = color; snack.show = true
}

async function loadData() {
  loading.value = true
  try {
    const [p, u, e] = await Promise.all([
      prestamoService.getAll(),
      usuarioService.getAll(),
      equipoService.getAll()
    ])
    prestamos.value = p.data
    usuarios.value  = u.data
    equipos.value   = e.data
  } catch { showSnack('Error al cargar datos', 'error') }
  finally { loading.value = false }
}

function openCreate() {
  isEdit.value = false
  Object.assign(form, { usuarioId: null, equipoId: null, fechaInicio: new Date().toISOString().split('T')[0], fechaFin: '', estado: 'ACTIVO' })
  dialog.value = true
}

function openEdit(item) {
  isEdit.value = true
  selected.value = item
  Object.assign(form, {
    usuarioId: item.usuario?.id,
    equipoId: item.equipo?.id,
    fechaInicio: item.fechaInicio,
    fechaFin: item.fechaFin,
    estado: item.estado
  })
  dialog.value = true
}

function openDelete(item) {
  selected.value = item
  deleteDialog.value = true
}

async function handleSave() {
  saving.value = true
  try {
    const payload = {
      usuario: { id: form.usuarioId },
      equipo:  { id: form.equipoId },
      fechaInicio: form.fechaInicio,
      fechaFin: form.fechaFin || null,
      estado: form.estado
    }
    if (isEdit.value) {
      await prestamoService.update(selected.value.id, payload)
      showSnack('Préstamo actualizado')
    } else {
      await prestamoService.create(payload)
      showSnack('Préstamo creado')
    }
    dialog.value = false
    await loadData()
  } catch (e) {
    showSnack(e.response?.data?.message || 'Error al guardar préstamo', 'error')
  } finally { saving.value = false }
}

async function handleDelete() {
  saving.value = true
  try {
    await prestamoService.delete(selected.value.id)
    showSnack('Préstamo eliminado')
    deleteDialog.value = false
    await loadData()
  } catch { showSnack('Error al eliminar', 'error') }
  finally { saving.value = false }
}

onMounted(loadData)
</script>

<style scoped>
.data-card { border: 1px solid rgba(255,255,255,0.06) !important; }
</style>
