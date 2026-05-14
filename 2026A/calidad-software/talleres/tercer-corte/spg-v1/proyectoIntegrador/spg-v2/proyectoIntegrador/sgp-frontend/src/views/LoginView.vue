<template>
  <div class="login-bg">
    <!-- Animated grid lines -->
    <div class="grid-overlay" />

    <v-container class="fill-height" fluid>
      <v-row align="center" justify="center" class="fill-height">
        <v-col cols="12" sm="8" md="5" lg="4">

          <!-- Header -->
          <div class="text-center mb-8">
            <div class="brand-icon mb-4">
              <v-icon size="36" color="primary">mdi-flask-outline</v-icon>
            </div>
            <h1 class="brand-title mono">SGP<span class="text-primary">LAB</span></h1>
            <p class="brand-sub">Sistema de Gestión de Préstamos</p>
          </div>

          <!-- Card -->
          <v-card class="login-card pa-8" color="surface">
            <div class="card-accent" />

            <h2 class="text-h6 font-weight-medium mb-1">Iniciar sesión</h2>
            <p class="text-body-2 text-medium-emphasis mb-6">Accede con tu cuenta de laboratorio</p>

            <v-form @submit.prevent="handleLogin">
              <v-text-field
                v-model="form.email"
                label="Correo electrónico"
                prepend-inner-icon="mdi-email-outline"
                type="email"
                :error-messages="errors.email"
                class="mb-3"
                bg-color="surface-variant"
              />

              <v-text-field
                v-model="form.password"
                label="Contraseña"
                prepend-inner-icon="mdi-lock-outline"
                :append-inner-icon="showPass ? 'mdi-eye-off' : 'mdi-eye'"
                :type="showPass ? 'text' : 'password'"
                :error-messages="errors.password"
                class="mb-2"
                bg-color="surface-variant"
                @click:append-inner="showPass = !showPass"
              />

              <!-- Role selector (demo - simulates backend auth) -->
              <v-select
                v-model="form.rol"
                label="Rol (demo)"
                :items="roles"
                prepend-inner-icon="mdi-shield-account-outline"
                class="mb-6"
                bg-color="surface-variant"
                hint="En producción esto lo determina el backend"
                persistent-hint
              />

              <v-btn
                type="submit"
                color="primary"
                size="large"
                block
                :loading="loading"
                class="font-weight-semibold"
              >
                <v-icon start>mdi-login</v-icon>
                Ingresar al sistema
              </v-btn>
            </v-form>

            <v-alert
              v-if="errorMsg"
              type="error"
              variant="tonal"
              class="mt-4"
              density="compact"
            >
              {{ errorMsg }}
            </v-alert>
          </v-card>

          <p class="text-center text-caption text-medium-emphasis mt-6 mono">
            SGP LAB v1.0 · {{ new Date().getFullYear() }}
          </p>
        </v-col>
      </v-row>
    </v-container>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuth } from '@/composables/useAuth'
import { usuarioService } from '@/services/api'

const router = useRouter()
const route = useRoute()
const { login } = useAuth()

const showPass = ref(false)
const loading = ref(false)
const errorMsg = ref('')

const roles = ['ADMINISTRADOR', 'CLIENTE']

const form = reactive({
  email: '',
  password: '',
  rol: 'ADMINISTRADOR'
})

const errors = reactive({ email: '', password: '' })

function validate() {
  errors.email = ''
  errors.password = ''
  let ok = true
  if (!form.email || !form.email.includes('@')) {
    errors.email = 'Ingresa un email válido'
    ok = false
  }
  if (!form.password || form.password.length < 3) {
    errors.password = 'Contraseña requerida'
    ok = false
  }
  return ok
}

async function handleLogin() {
  if (!validate()) return
  loading.value = true
  errorMsg.value = ''

  try {
    // Demo: busca usuario por email en el backend real
    // En producción aquí iría un endpoint /api/auth/login
    const res = await usuarioService.getAll()
    const usuarios = res.data
    const found = usuarios.find(u => u.email === form.email)

    if (found) {
      login({ ...found, rol: form.rol })
    } else {
      // Modo demo: crea sesión simulada si no hay backend
      login({
        id: 0,
        nombre: form.email.split('@')[0],
        email: form.email,
        rol: form.rol,
        estado: 'ACTIVO'
      })
    }

    const redirect = route.query.redirect || '/dashboard'
    router.push(redirect)
  } catch {
    // Si no hay backend, modo demo puro
    login({
      id: 0,
      nombre: form.email.split('@')[0],
      email: form.email,
      rol: form.rol,
      estado: 'ACTIVO'
    })
    router.push('/dashboard')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-bg {
  min-height: 100vh;
  background: #0a0e14;
  position: relative;
  overflow: hidden;
}

.grid-overlay {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,212,255,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.04) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none;
}

.login-bg::before {
  content: '';
  position: absolute;
  top: -200px;
  right: -200px;
  width: 600px;
  height: 600px;
  background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
  pointer-events: none;
}

.brand-icon {
  width: 64px;
  height: 64px;
  background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,212,255,0.05));
  border: 1px solid rgba(0,212,255,0.3);
  border-radius: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.brand-title {
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: 4px;
  color: #e2e8f0;
}

.brand-sub {
  color: rgba(226,232,240,0.5);
  font-size: 0.8rem;
  letter-spacing: 1px;
  margin-top: 4px;
}

.login-card {
  border: 1px solid rgba(255,255,255,0.07) !important;
  position: relative;
  overflow: hidden;
}

.card-accent {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, #00d4ff, transparent);
}
</style>
