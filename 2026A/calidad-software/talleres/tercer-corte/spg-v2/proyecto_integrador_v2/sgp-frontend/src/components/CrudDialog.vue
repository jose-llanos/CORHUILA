<template>
  <v-dialog v-model="model" :max-width="maxWidth" persistent>
    <v-card color="surface" class="crud-dialog">
      <div class="dialog-accent" />

      <v-card-title class="d-flex align-center pa-6 pb-4">
        <v-icon :color="color" class="mr-2">{{ icon }}</v-icon>
        <span class="text-h6 font-weight-medium">{{ title }}</span>
        <v-spacer />
        <v-btn icon="mdi-close" variant="text" size="small" @click="model = false" />
      </v-card-title>

      <v-divider class="border-opacity-10" />

      <v-card-text class="pa-6">
        <slot />
      </v-card-text>

      <v-divider class="border-opacity-10" />

      <v-card-actions class="pa-4">
        <v-spacer />
        <v-btn variant="text" @click="model = false">Cancelar</v-btn>
        <v-btn
          :color="color"
          :loading="loading"
          variant="flat"
          @click="$emit('confirm')"
        >
          {{ confirmLabel }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup>
defineProps({
  title: String,
  icon: { type: String, default: 'mdi-pencil' },
  color: { type: String, default: 'primary' },
  confirmLabel: { type: String, default: 'Guardar' },
  maxWidth: { type: [String, Number], default: 560 },
  loading: Boolean
})
defineEmits(['confirm'])
const model = defineModel()
</script>

<style scoped>
.crud-dialog {
  border: 1px solid rgba(255,255,255,0.07) !important;
  position: relative;
  overflow: hidden;
}
.dialog-accent {
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, #00d4ff, transparent);
}
</style>
