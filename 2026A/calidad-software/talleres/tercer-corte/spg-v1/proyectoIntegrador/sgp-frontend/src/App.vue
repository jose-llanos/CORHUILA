<template>
  <v-app>
    <router-view />
  </v-app>
</template>

<script setup>
</script>

<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --font-mono: 'Space Mono', monospace;
  --font-body: 'DM Sans', sans-serif;
}

body {
  font-family: var(--font-body);
  background: #0a0e14;
}

.v-application {
  font-family: var(--font-body) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #111820; }
::-webkit-scrollbar-thumb { background: #2a3a4a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00d4ff55; }

/* Mono labels */
.mono { font-family: var(--font-mono) !important; }

/* Glow effects */
.glow-primary { box-shadow: 0 0 20px rgba(0, 212, 255, 0.15) !important; }
.glow-error { box-shadow: 0 0 20px rgba(239, 68, 68, 0.15) !important; }
.glow-success { box-shadow: 0 0 20px rgba(16, 185, 129, 0.15) !important; }

/* Table tweaks */
.v-data-table { background: transparent !important; }
.v-data-table .v-data-table__td { border-bottom: 1px solid rgba(255,255,255,0.05) !important; }
</style>
