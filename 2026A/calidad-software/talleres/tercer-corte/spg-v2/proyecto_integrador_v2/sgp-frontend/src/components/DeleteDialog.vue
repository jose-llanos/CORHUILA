<template>
  <v-dialog v-model="model" max-width="400" persistent>
    <v-card color="surface" class="pa-2">
      <v-card-title class="d-flex align-center pa-5">
        <v-icon color="error" class="mr-2">mdi-trash-can-outline</v-icon>
        <span class="text-h6">Confirmar eliminación</span>
      </v-card-title>
      <v-card-text class="px-5 pb-2">
        <p class="text-body-2 text-medium-emphasis">
          ¿Estás seguro de que deseas eliminar <strong class="text-white">{{ itemName }}</strong>?
          Esta acción no se puede deshacer.
        </p>
      </v-card-text>
      <v-card-actions class="pa-4">
        <v-spacer />
        <v-btn variant="text" @click="model = false">Cancelar</v-btn>
        <v-btn color="error" variant="flat" :loading="loading" @click="$emit('confirm')">
          Eliminar
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup>
defineProps({ itemName: String, loading: Boolean })
defineEmits(['confirm'])
const model = defineModel()
</script>
