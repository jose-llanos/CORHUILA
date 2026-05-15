<template>
  <div class="sin-acceso-bg fill-height d-flex align-center justify-center">
    <div class="text-center pa-8">
      <div class="error-code mono">403</div>
      <v-icon size="64" color="error" class="mb-4">mdi-shield-lock-outline</v-icon>
      <h2 class="text-h5 mb-2">Acceso denegado</h2>
      <p class="text-medium-emphasis mb-6">No tienes permisos para acceder a esta sección.</p>
      <v-btn to="/dashboard" color="primary" variant="flat">
        <v-icon start>mdi-home</v-icon>
        Ir al dashboard
      </v-btn>
    </div>
  </div>
</template>

<style scoped>
.sin-acceso-bg {
  min-height: 100vh;
  background: #0a0e14;
}
.error-code {
  font-size: 6rem;
  font-weight: 700;
  color: rgba(239,68,68,0.15);
  line-height: 1;
  margin-bottom: 16px;
}
</style>
