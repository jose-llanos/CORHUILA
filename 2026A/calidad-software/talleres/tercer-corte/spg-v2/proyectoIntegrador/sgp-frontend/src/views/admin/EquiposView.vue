<template>
  <div>
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
      <v-spacer />
      <v-btn color="secondary" prepend-icon="mdi-plus-box" @click="openCreate">
        Nuevo equipo
      </v-btn>
    </div>

    <v-card color="surface" class="data-card">
      <v-data-table
        :headers="headers"
        :items="equipos"
        :search="search"
        :loading="loading"
        hover
        no-data-text="No hay equipos registrados"
      >
        <template #item.cantidad="{ item }">
          <v-chip
            :color="item.cantidad > 5 ? 'success' : item.cantidad > 0 ? 'warning' : 'error'"
            size="small"
            variant="tonal"
            class="mono"
          >
            {{ item.cantidad }}
          </v-chip>
        </template>

        <template #item.estado="{ item }">
          <v-chip :color="estadoEquipoColor(item.estado)" size="small" variant="tonal">
            <v-icon start size="10">mdi-circle</v-icon>
            {{ item.estado }}
          </v-chip>
        </template>

        <template #item.codigoInventario="{ item }">
          <span class="mono text-primary" style="font-size:0.8rem">{{ item.codigoInventario }}</span>
        </template>

        <template #item.actions="{ item }">
          <div class="d-flex gap-1">
            <v-btn icon="mdi-pencil-outline" variant="text" size="small" color="secondary" @click="openEdit(item)" />
            <v-btn icon="mdi-trash-can-outline" variant="text" size="small" color="error" @click="openDelete(item)" />
          </div>
        </template>
      </v-data-table>
    </v-card>

    <CrudDialog
      v-model="dialog"
      :title="isEdit ? 'Editar equipo' : 'Nuevo equipo'"
      icon="mdi-desktop-classic"
      color="secondary"
      :loading="saving"
      @confirm="handleSave"
    >
      <v-row dense>
        <v-col cols="12">
          <v-text-field v-model="form.nombre" label="Nombre del equipo" prepend-inner-icon="mdi-tag" bg-color="surface-variant" />
        </v-col>
        <v-col cols="12">
          <v-text-field v-model="form.codigoInventario" label="Código de inventario" prepend-inner-icon="mdi-barcode" class="mono" bg-color="surface-variant" />
        </v-col>
        <v-col cols="6">
          <v-text-field v-model.number="form.cantidad" label="Cantidad" type="number" prepend-inner-icon="mdi-counter" bg-color="surface-variant" />
        </v-col>
        <v-col cols="6">
          <v-select
            v-model="form.estado"
            label="Estado"
            :items="['DISPONIBLE','PRESTADO','EN_MANTENIMIENTO']"
            prepend-inner-icon="mdi-toggle-switch"
            bg-color="surface-variant"
          />
        </v-col>
      </v-row>
    </CrudDialog>

    <DeleteDialog
      v-model="deleteDialog"
      :item-name="selected?.nombre"
      :loading="saving"
      @confirm="handleDelete"
    />

    <v-snackbar v-model="snack.show" :color="snack.color" location="bottom right" rounded="lg">
      {{ snack.text }}
    </v-snackbar>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { equipoService } from '@/services/api'
import CrudDialog from '@/components/CrudDialog.vue'
import DeleteDialog from '@/components/DeleteDialog.vue'

const headers = [
  { title: 'ID',     key: 'id',     width: 60 },
  { title: 'Código', key: 'codigoInventario' },
  { title: 'Nombre', key: 'nombre', sortable: true },
  { title: 'Stock',  key: 'cantidad', sortable: true },
  { title: 'Estado', key: 'estado' },
  { title: 'Acciones', key: 'actions', sortable: false, align: 'end' },
]

const equipos = ref([])
const loading = ref(false)
const saving  = ref(false)
const search  = ref('')
const dialog  = ref(false)
const deleteDialog = ref(false)
const isEdit  = ref(false)
const selected = ref(null)

const form = reactive({ nombre: '', codigoInventario: '', cantidad: 1, estado: 'DISPONIBLE' })
const snack = reactive({ show: false, text: '', color: 'success' })

function estadoEquipoColor(e) {
  return { DISPONIBLE: 'success', PRESTADO: 'warning', EN_MANTENIMIENTO: 'error' }[e] || 'info'
}
function showSnack(text, color = 'success') {
  snack.text = text; snack.color = color; snack.show = true
}

async function loadData() {
  loading.value = true
  try {
    const res = await equipoService.getAll()
    equipos.value = res.data
  } catch { showSnack('Error al cargar equipos', 'error') }
  finally { loading.value = false }
}

function openCreate() {
  isEdit.value = false
  Object.assign(form, { nombre: '', codigoInventario: '', cantidad: 1, estado: 'DISPONIBLE' })
  dialog.value = true
}

function openEdit(item) {
  isEdit.value = true
  selected.value = item
  Object.assign(form, { nombre: item.nombre, codigoInventario: item.codigoInventario, cantidad: item.cantidad, estado: item.estado })
  dialog.value = true
}

function openDelete(item) {
  selected.value = item
  deleteDialog.value = true
}

async function handleSave() {
  saving.value = true
  try {
    if (isEdit.value) {
      await equipoService.update(selected.value.id, form)
      showSnack('Equipo actualizado')
    } else {
      await equipoService.create(form)
      showSnack('Equipo creado')
    }
    dialog.value = false
    await loadData()
  } catch { showSnack('Error al guardar equipo', 'error') }
  finally { saving.value = false }
}

async function handleDelete() {
  saving.value = true
  try {
    await equipoService.delete(selected.value.id)
    showSnack('Equipo eliminado')
    deleteDialog.value = false
    await loadData()
  } catch { showSnack('Error al eliminar equipo', 'error') }
  finally { saving.value = false }
}

onMounted(loadData)
</script>

<style scoped>
.data-card { border: 1px solid rgba(255,255,255,0.06) !important; }
</style>
