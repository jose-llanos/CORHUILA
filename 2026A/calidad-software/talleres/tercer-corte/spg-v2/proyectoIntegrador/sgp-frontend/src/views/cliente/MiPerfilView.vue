<template>
  <div>
    <v-row justify="center">
      <v-col cols="12" md="7" lg="6">

        <v-card color="surface" class="profile-card pa-6 mb-5 text-center">
          <div class="avatar-ring mx-auto mb-4">
            <v-avatar size="80" color="primary">
              <span class="text-h4 font-weight-bold text-black">{{ inicial }}</span>
            </v-avatar>
          </div>
          <h2 class="text-h6 font-weight-semibold">{{ currentUser?.nombre }}</h2>
          <p class="text-caption text-medium-emphasis mono mb-3">{{ currentUser?.email }}</p>
          <div class="d-flex justify-center gap-2">
            <v-chip :color="currentUser?.rol === 'ADMINISTRADOR' ? 'primary' : 'secondary'" size="small" variant="tonal">
              <v-icon start size="12">{{ currentUser?.rol === 'ADMINISTRADOR' ? 'mdi-shield' : 'mdi-account' }}</v-icon>
              {{ currentUser?.rol }}
            </v-chip>
            <v-chip color="success" size="small" variant="tonal">
              <v-icon start size="10">mdi-circle</v-icon>
              {{ currentUser?.estado }}
            </v-chip>
          </div>
        </v-card>

        <v-card color="surface" class="profile-card pa-6">
          <div class="d-flex align-center mb-5">
            <v-icon color="primary" class="mr-2">mdi-account-edit-outline</v-icon>
            <h3 class="text-h6 font-weight-medium">Editar datos</h3>
          </div>

          <v-form @submit.prevent="handleSave">
            <v-text-field
              v-model="form.nombre"
              label="Nombre completo"
              prepend-inner-icon="mdi-account-outline"
              bg-color="surface-variant"
              class="mb-3"
            />
            <v-text-field
              v-model="form.email"
              label="Correo electrónico"
              prepend-inner-icon="mdi-email-outline"
              type="email"
              bg-color="surface-variant"
              class="mb-5"
            />
            <v-alert type="info" variant="tonal" density="compact" class="mb-5">
              El rol y estado solo pueden ser modificados por un administrador.
            </v-alert>
            <v-btn type="submit" color="primary" block :loading="saving" prepend-icon="mdi-content-save-outline">
              Guardar cambios
            </v-btn>
          </v-form>
        </v-card>

      </v-col>
    </v-row>

    <v-snackbar v-model="snack.show" :color="snack.color" location="bottom right" rounded="lg">
      {{ snack.text }}
    </v-snackbar>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { usuarioService } from '@/services/api'
import { useAuth } from '@/composables/useAuth'

const { currentUser, login } = useAuth()
const saving = ref(false)
const snack  = reactive({ show: false, text: '', color: 'success' })
const form   = reactive({ nombre: '', email: '' })
const inicial = computed(() => currentUser.value?.nombre?.charAt(0).toUpperCase() || 'U')

async function handleSave() {
  saving.value = true
  try {
    const res = await usuarioService.update(currentUser.value.id, { nombre: form.nombre, email: form.email })
    login({ ...currentUser.value, nombre: res.data.nombre, email: res.data.email })
    snack.text = 'Perfil actualizado'; snack.color = 'success'; snack.show = true
  } catch {
    snack.text = 'Error al actualizar'; snack.color = 'error'; snack.show = true
  } finally { saving.value = false }
}

onMounted(() => {
  form.nombre = currentUser.value?.nombre || ''
  form.email  = currentUser.value?.email  || ''
})
</script>

<style scoped>
.profile-card { border: 1px solid rgba(255,255,255,0.06) !important; }
.avatar-ring {
  width: 92px; height: 92px;
  border-radius: 50%;
  background: linear-gradient(135deg, rgba(0,212,255,0.3), rgba(124,58,237,0.3));
  display: flex; align-items: center; justify-content: center;
}
</style>
