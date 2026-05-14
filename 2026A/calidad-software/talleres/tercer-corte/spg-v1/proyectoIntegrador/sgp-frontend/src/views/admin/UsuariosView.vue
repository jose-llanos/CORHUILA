<template>
  <div>
    <!-- Toolbar -->
    <div class="d-flex align-center mb-5 gap-3 flex-wrap">
      <v-text-field
        v-model="search"
        prepend-inner-icon="mdi-magnify"
        placeholder="Buscar usuario..."
        variant="outlined"
        density="compact"
        bg-color="surface"
        hide-details
        style="max-width:280px"
      />
      <v-spacer />
      <v-btn color="primary" prepend-icon="mdi-account-plus" @click="openCreate">
        Nuevo usuario
      </v-btn>
    </div>

    <!-- Table -->
    <v-card color="surface" class="data-card">
      <v-data-table
        :headers="headers"
        :items="usuarios"
        :search="search"
        :loading="loading"
        loading-text="Cargando usuarios..."
        no-data-text="No hay usuarios registrados"
        hover
      >
        <!-- Estado chip -->
        <template #item.estado="{ item }">
          <v-chip
            :color="item.estado === 'ACTIVO' ? 'success' : 'error'"
            size="small"
            variant="tonal"
            class="mono"
          >
            {{ item.estado }}
          </v-chip>
        </template>

        <!-- Rol chip -->
        <template #item.rol="{ item }">
          <v-chip
            :color="item.rol === 'ADMINISTRADOR' ? 'primary' : 'secondary'"
            size="small"
            variant="tonal"
          >
            <v-icon start size="12">{{ item.rol === 'ADMINISTRADOR' ? 'mdi-shield' : 'mdi-account' }}</v-icon>
            {{ item.rol || '—' }}
          </v-chip>
        </template>

        <!-- Actions -->
        <template #item.actions="{ item }">
          <div class="d-flex gap-1">
            <v-btn icon="mdi-pencil-outline" variant="text" size="small" color="primary" @click="openEdit(item)" />
            <v-btn icon="mdi-trash-can-outline" variant="text" size="small" color="error" @click="openDelete(item)" />
          </div>
        </template>
      </v-data-table>
    </v-card>

    <!-- Create/Edit Dialog -->
    <CrudDialog
      v-model="dialog"
      :title="isEdit ? 'Editar usuario' : 'Nuevo usuario'"
      icon="mdi-account-outline"
      :loading="saving"
      @confirm="handleSave"
    >
      <v-row dense>
        <v-col cols="12">
          <v-text-field v-model="form.nombre" label="Nombre completo" prepend-inner-icon="mdi-account" bg-color="surface-variant" />
        </v-col>
        <v-col cols="12">
          <v-text-field v-model="form.email" label="Correo electrónico" prepend-inner-icon="mdi-email" type="email" bg-color="surface-variant" />
        </v-col>
        <v-col cols="6">
          <v-select
            v-model="form.rol"
            label="Rol"
            :items="['ADMINISTRADOR','CLIENTE']"
            prepend-inner-icon="mdi-shield-account"
            bg-color="surface-variant"
          />
        </v-col>
        <v-col cols="6">
          <v-select
            v-model="form.estado"
            label="Estado"
            :items="['ACTIVO','PENALIZADO']"
            prepend-inner-icon="mdi-toggle-switch"
            bg-color="surface-variant"
          />
        </v-col>
      </v-row>
    </CrudDialog>

    <!-- Delete Dialog -->
    <DeleteDialog
      v-model="deleteDialog"
      :item-name="selected?.nombre"
      :loading="saving"
      @confirm="handleDelete"
    />

    <!-- Snackbar -->
    <v-snackbar v-model="snack.show" :color="snack.color" location="bottom right" rounded="lg">
      {{ snack.text }}
    </v-snackbar>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { usuarioService } from '@/services/api'
import CrudDialog from '@/components/CrudDialog.vue'
import DeleteDialog from '@/components/DeleteDialog.vue'

const headers = [
  { title: 'ID',     key: 'id',     width: 60 },
  { title: 'Nombre', key: 'nombre', sortable: true },
  { title: 'Email',  key: 'email',  sortable: true },
  { title: 'Rol',    key: 'rol' },
  { title: 'Estado', key: 'estado' },
  { title: 'Acciones', key: 'actions', sortable: false, align: 'end' },
]

const usuarios = ref([])
const loading = ref(false)
const saving  = ref(false)
const search  = ref('')
const dialog  = ref(false)
const deleteDialog = ref(false)
const isEdit  = ref(false)
const selected = ref(null)

const form = reactive({ nombre: '', email: '', rol: 'CLIENTE', estado: 'ACTIVO' })
const snack = reactive({ show: false, text: '', color: 'success' })

function showSnack(text, color = 'success') {
  snack.text = text; snack.color = color; snack.show = true
}

async function loadData() {
  loading.value = true
  try {
    const res = await usuarioService.getAll()
    usuarios.value = res.data
  } catch { showSnack('Error al cargar usuarios', 'error') }
  finally { loading.value = false }
}

function openCreate() {
  isEdit.value = false
  Object.assign(form, { nombre: '', email: '', rol: 'CLIENTE', estado: 'ACTIVO' })
  dialog.value = true
}

function openEdit(item) {
  isEdit.value = true
  selected.value = item
  Object.assign(form, { nombre: item.nombre, email: item.email, rol: item.rol, estado: item.estado })
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
      await usuarioService.update(selected.value.id, form)
      showSnack('Usuario actualizado correctamente')
    } else {
      await usuarioService.create(form)
      showSnack('Usuario creado correctamente')
    }
    dialog.value = false
    await loadData()
  } catch { showSnack('Error al guardar usuario', 'error') }
  finally { saving.value = false }
}

async function handleDelete() {
  saving.value = true
  try {
    await usuarioService.delete(selected.value.id)
    showSnack('Usuario eliminado')
    deleteDialog.value = false
    await loadData()
  } catch { showSnack('Error al eliminar usuario', 'error') }
  finally { saving.value = false }
}

onMounted(loadData)
</script>

<style scoped>
.data-card { border: 1px solid rgba(255,255,255,0.06) !important; }
</style>
