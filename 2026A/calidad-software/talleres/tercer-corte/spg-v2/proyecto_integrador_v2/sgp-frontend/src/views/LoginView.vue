<template>
  <v-container class="fill-height" fluid>
    <v-row align="center" justify="center">
      <v-col cols="12" sm="8" md="5" lg="4">
        <v-card elevation="10" class="pa-4">
          <v-card-title class="text-h5 text-center">
            SGP LAB
          </v-card-title>
          <v-card-subtitle class="text-center mb-2">
            Sistema de Gestion de Prestamos
          </v-card-subtitle>

          <v-card-text>
            <v-alert
              v-if="sessionExpired"
              type="warning"
              variant="tonal"
              density="compact"
              closable
              class="mb-3"
              @click:close="sessionExpired = false"
            >
              Tu sesion expiro. Por favor, ingresa nuevamente.
            </v-alert>

            <v-alert
              v-if="errorMessage"
              type="error"
              variant="tonal"
              density="compact"
              closable
              class="mb-3"
              @click:close="errorMessage = ''"
            >
              {{ errorMessage }}
            </v-alert>

            <v-form ref="formRef" v-model="formValid" @submit.prevent="handleLogin">
              <v-text-field
                v-model="email"
                label="Correo electronico"
                type="email"
                autocomplete="username"
                prepend-inner-icon="mdi-email-outline"
                :rules="emailRules"
                required
                :disabled="loading"
              />

              <v-text-field
                v-model="password"
                label="Contrasena"
                :type="showPassword ? 'text' : 'password'"
                autocomplete="current-password"
                prepend-inner-icon="mdi-lock-outline"
                :append-inner-icon="showPassword ? 'mdi-eye-off' : 'mdi-eye'"
                @click:append-inner="showPassword = !showPassword"
                :rules="passwordRules"
                required
                :disabled="loading"
              />

              <v-btn
                type="submit"
                color="primary"
                block
                size="large"
                :loading="loading"
                :disabled="!formValid || loading"
                class="mt-2"
              >
                Ingresar
              </v-btn>
            </v-form>
          </v-card-text>

          <v-card-actions class="justify-center">
            <small class="text-medium-emphasis">
              Si no tienes cuenta, solicitala al administrador del laboratorio.
            </small>
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { authApi, getErrorMessage } from '@/services/api'
import { useAuth } from '@/composables/useAuth'

const router = useRouter()
const route = useRoute()
const { setSession, isAuthenticated, role } = useAuth()

const email = ref('')
const password = ref('')
const showPassword = ref(false)
const loading = ref(false)
const errorMessage = ref('')
const sessionExpired = ref(false)
const formValid = ref(false)
const formRef = ref(null)

const emailRules = [
  (v) => !!v || 'El email es obligatorio',
  (v) => /.+@.+\..+/.test(v) || 'Formato de email invalido'
]
const passwordRules = [
  (v) => !!v || 'La contrasena es obligatoria',
  (v) => (v && v.length >= 6) || 'Minimo 6 caracteres'
]

onMounted(() => {
  if (route.query.expired === '1') {
    sessionExpired.value = true
  }
  if (isAuthenticated.value) {
    redirectByRole()
  }
})

function redirectByRole() {
  if (role.value === 'ADMINISTRADOR') {
    router.replace('/admin')
  } else {
    router.replace('/catalogo')
  }
}

async function handleLogin() {
  const { valid } = await formRef.value.validate()
  if (!valid) return

  errorMessage.value = ''
  sessionExpired.value = false
  loading.value = true

  try {
    const response = await authApi.login(email.value, password.value)
    setSession(response)
    redirectByRole()
  } catch (err) {
    errorMessage.value = getErrorMessage(err) || 'Credenciales invalidas'
  } finally {
    loading.value = false
  }
}
</script>
