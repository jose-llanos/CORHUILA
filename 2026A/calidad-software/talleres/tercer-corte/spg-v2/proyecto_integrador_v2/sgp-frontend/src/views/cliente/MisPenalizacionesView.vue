<template>
  <div>
    <!-- Estado general -->
    <v-alert
      v-if="tienePenalizacionActiva"
      type="error"
      variant="tonal"
      border="start"
      prominent
      class="mb-5"
    >
      <template #title><span class="mono">CUENTA RESTRINGIDA</span></template>
      Tienes una penalización activa. No podrás solicitar nuevos préstamos hasta que sea levantada o cumplida.
    </v-alert>

    <v-alert
      v-else-if="!loading && penalizaciones.length > 0"
      type="success"
      variant="tonal"
      border="start"
      class="mb-5"
    >
      No tienes penalizaciones activas. Tu cuenta está al día.
    </v-alert>

    <!-- Cards de penalizaciones -->
    <div v-if="loading" class="d-flex justify-center py-10">
      <v-progress-circular indeterminate color="error" />
    </div>

    <v-row v-else-if="penalizaciones.length > 0">
      <v-col v-for="pen in penalizaciones" :key="pen.id" cols="12" md="6">
        <v-card color="surface" class="pen-card" :class="`pen-card--${estadoColor(pen.estado)}`">
          <v-card-text class="pa-5">
            <div class="d-flex align-start justify-space-between mb-4">
              <div class="pen-icon" :class="`pen-icon--${estadoColor(pen.estado)}`">
                <v-icon :color="estadoColor(pen.estado)" size="20">mdi-shield-alert-outline</v-icon>
              </div>
              <v-chip :color="estadoColor(pen.estado)" size="small" variant="tonal" class="mono">
                {{ pen.estado }}
              </v-chip>
            </div>

            <p class="text-body-2 mb-4" style="line-height:1.6">
              <v-icon size="14" color="medium-emphasis" class="mr-1">mdi-text</v-icon>
              {{ pen.motivo }}
            </p>

            <v-divider class="border-opacity-10 mb-4" />

            <div class="d-flex gap-4">
              <div>
                <div class="text-caption text-medium-emphasis">Inicio</div>
                <div class="mono text-body-2">{{ pen.fechaInicio }}</div>
              </div>
              <div>
                <div class="text-caption text-medium-emphasis">Fin</div>
                <div class="mono text-body-2">{{ pen.fechaFin }}</div>
              </div>
              <div v-if="pen.estado === 'ACTIVA'">
                <div class="text-caption text-medium-emphasis">Días restantes</div>
                <div class="mono text-body-2 text-error font-weight-bold">{{ diasRestantes(pen.fechaFin) }}</div>
              </div>
            </div>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <!-- Sin penalizaciones -->
    <v-card v-else color="surface" class="pa-10 text-center">
      <v-icon size="64" color="success" class="mb-4">mdi-shield-check-outline</v-icon>
      <h3 class="text-h6 mb-2">Sin penalizaciones</h3>
      <p class="text-medium-emphasis">Tu historial está limpio. ¡Sigue así!</p>
    </v-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { penalizacionService } from '@/services/api'
import { useAuth } from '@/composables/useAuth'

const { currentUser } = useAuth()

const penalizaciones = ref([])
const loading        = ref(false)

const tienePenalizacionActiva = computed(() =>
  penalizaciones.value.some(p => p.estado === 'ACTIVA')
)

function estadoColor(e) {
  return { ACTIVA:'error', CUMPLIDA:'success', LEVANTADA:'info' }[e] || 'default'
}

function diasRestantes(fechaFin) {
  if (!fechaFin) return '—'
  const diff = Math.ceil((new Date(fechaFin) - new Date()) / (1000 * 60 * 60 * 24))
  return diff > 0 ? `${diff} días` : 'Vencida'
}

async function loadData() {
  loading.value = true
  try {
    const res = await penalizacionService.getAll()
    penalizaciones.value = currentUser.value?.id === 0
      ? res.data
      : res.data.filter(p => p.usuario?.id === currentUser.value?.id)
  } finally { loading.value = false }
}

onMounted(loadData)
</script>

<style scoped>
.pen-card {
  border: 1px solid rgba(255,255,255,0.06) !important;
  transition: transform 0.2s;
}
.pen-card:hover { transform: translateY(-2px); }
.pen-card--error   { border-color: rgba(239,68,68,0.2) !important; }
.pen-card--success { border-color: rgba(16,185,129,0.2) !important; }
.pen-card--info    { border-color: rgba(59,130,246,0.2) !important; }

.pen-icon {
  width: 40px; height: 40px;
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
}
.pen-icon--error   { background: rgba(239,68,68,0.1); }
.pen-icon--success { background: rgba(16,185,129,0.1); }
.pen-icon--info    { background: rgba(59,130,246,0.1); }
</style>
