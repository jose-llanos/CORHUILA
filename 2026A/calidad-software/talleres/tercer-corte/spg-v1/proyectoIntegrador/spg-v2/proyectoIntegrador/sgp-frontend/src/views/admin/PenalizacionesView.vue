<template>
  <div>
    <div class="d-flex align-center mb-5 gap-3 flex-wrap">
      <v-text-field
        v-model="search"
        prepend-inner-icon="mdi-magnify"
        placeholder="Buscar penalización..."
        variant="outlined"
        density="compact"
        bg-color="surface"
        hide-details
        style="max-width:280px"
      />
      <v-spacer />
      <v-btn color="error" prepend-icon="mdi-shield-alert" @click="openCreate">
        Nueva penalización
      </v-btn>
    </div>

    <v-card color="surface" class="data-card">
      <v-data-table
        :headers="headers"
        :items="penalizaciones"
        :search="search"
        :loading="loading"
        hover
        no-data-text="No hay penalizaciones registradas"
      >
        <template #item.estado="{ item }">
          <v-chip :color="estadoColor(item.estado)" size="small" variant="tonal" class="mono">
            <v-icon start size="10">mdi-circle</v-icon>
            {{ item.estado }}
          </v-chip>
        </template>

        <template #item.usuario="{ item }">
          <div class="d-flex align-center">
            <v-avatar size="22" color="error" class="mr-2">
              <span style="font-size:10px;color:white;font-weight:700">{{ item.usuario?.nombre?.charAt(0) || 'U' }}</span>
            </v-avatar>
            <span>{{ item.usuario?.nombre || `ID: ${item.usuario?.id}` }}</span>
          </div>
        </template>

        <template #item.motivo="{ item }">
          <span class="text-medium-emphasis text-caption">{{ item.motivo }}</span>
        </template>

        <template #item.actions="{ item }">
          <div class="d-flex gap-1">
            <v-btn icon="mdi-pencil-outline" variant="text" size="small" color="error" @click="openEdit(item)" />
            <v-btn icon="mdi-trash-can-outline" variant="text" size="small" color="error" @click="openDelete(item)" />
          </div>
        </template>
      </v-data-table>
    </v-card>

    <CrudDialog
      v-model="dialog"
      :title="isEdit ? 'Editar penalización' : 'Nueva penalización'"
      icon="mdi-shield-alert-outline"
      color="error"
      :loading="saving"
      @confirm="handleSave"
    >
      <v-row dense>
        <v-col cols="12">
          <v-select
            v-model="form.usuarioId"
            label="Usuario penalizado"
            :items="usuarios"
            item-title="nombre"
            item-value="id"
            prepend-inner-icon="mdi-account-alert"
            bg-color="surface-variant"
          />
        </v-col>
        <v-col cols="12">
          <v-textarea
            v-model="form.motivo"
            label="Motivo de la penalización"
            prepend-inner-icon="mdi-text"
            bg-color="surface-variant"
            rows="3"
            variant="outlined"
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
            :items="['ACTIVA','CUMPLIDA','LEVANTADA']"
            prepend-inner-icon="mdi-toggle-switch"
            bg-color="surface-variant"
          />
        </v-col>
      </v-row>
    </CrudDialog>

    <DeleteDialog
      v-model="deleteDialog"
      :item-name="`Penalización #${selected?.id}`"
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
import { penalizacionService, usuarioService } from '@/services/api'
import CrudDialog from '@/components/CrudDialog.vue'
import DeleteDialog from '@/components/DeleteDialog.vue'

const headers = [
  { title: 'ID',          key: 'id',          width: 60 },
  { title: 'Usuario',     key: 'usuario',     sortable: false },
  { title: 'Motivo',      key: 'motivo' },
  { title: 'Inicio',      key: 'fechaInicio' },
  { title: 'Fin',         key: 'fechaFin' },
  { title: 'Estado',      key: 'estado' },
  { title: 'Acciones',    key: 'actions', sortable: false, align: 'end' },
]

const penalizaciones = ref([])
const usuarios       = ref([])
const loading        = ref(false)
const saving         = ref(false)
const search         = ref('')
const dialog         = ref(false)
const deleteDialog   = ref(false)
const isEdit         = ref(false)
const selected       = ref(null)

const form = reactive({ usuarioId: null, motivo: '', fechaInicio: '', fechaFin: '', estado: 'ACTIVA' })
const snack = reactive({ show: false, text: '', color: 'success' })

function estadoColor(e) {
  return { ACTIVA:'error', CUMPLIDA:'success', LEVANTADA:'info' }[e] || 'default'
}
function showSnack(text, color = 'success') {
  snack.text = text; snack.color = color; snack.show = true
}

async function loadData() {
  loading.value = true
  try {
    const [p, u] = await Promise.all([penalizacionService.getAll(), usuarioService.getAll()])
    penalizaciones.value = p.data
    usuarios.value = u.data
  } catch { showSnack('Error al cargar datos', 'error') }
  finally { loading.value = false }
}

function openCreate() {
  isEdit.value = false
  Object.assign(form, { usuarioId: null, motivo: '', fechaInicio: new Date().toISOString().split('T')[0], fechaFin: '', estado: 'ACTIVA' })
  dialog.value = true
}

function openEdit(item) {
  isEdit.value = true
  selected.value = item
  Object.assign(form, {
    usuarioId: item.usuario?.id,
    motivo: item.motivo,
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
      motivo: form.motivo,
      fechaInicio: form.fechaInicio,
      fechaFin: form.fechaFin,
      estado: form.estado
    }
    if (isEdit.value) {
      await penalizacionService.update(selected.value.id, payload)
      showSnack('Penalización actualizada')
    } else {
      await penalizacionService.create(payload)
      showSnack('Penalización creada')
    }
    dialog.value = false
    await loadData()
  } catch { showSnack('Error al guardar penalización', 'error') }
  finally { saving.value = false }
}

async function handleDelete() {
  saving.value = true
  try {
    await penalizacionService.delete(selected.value.id)
    showSnack('Penalización eliminada')
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
