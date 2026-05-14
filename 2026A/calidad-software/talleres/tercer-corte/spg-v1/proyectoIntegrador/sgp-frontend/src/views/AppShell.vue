<template>
  <v-layout class="app-layout">

    <!-- ── Sidebar ─────────────────────────────────────────── -->
    <v-navigation-drawer
      v-model="drawer"
      permanent
      :rail="rail"
      color="surface"
      class="sidebar"
      width="240"
      rail-width="64"
    >
      <!-- Brand / toggle -->
      <div class="sidebar-brand" @click="rail = !rail">
        <div class="brand-dot">
          <v-icon color="primary" :size="22">mdi-flask-outline</v-icon>
        </div>
        <Transition name="fade">
          <span v-if="!rail" class="brand-name mono">SGP<span class="text-primary">LAB</span></span>
        </Transition>
        <v-spacer />
        <v-icon v-if="!rail" size="18" color="medium-emphasis">mdi-chevron-left</v-icon>
      </div>

      <v-divider class="border-opacity-10 mb-2" />

      <!-- User chip (solo cuando está expandido) -->
      <Transition name="fade">
        <div v-if="!rail" class="px-3 mb-3">
          <div class="user-chip">
            <v-avatar size="28" color="primary" class="mr-2">
              <span class="text-caption font-weight-bold text-black">{{ userInitial }}</span>
            </v-avatar>
            <div class="flex-grow-1 min-w-0">
              <div class="text-caption font-weight-medium text-truncate">{{ currentUser?.nombre }}</div>
              <div class="text-caption mono" style="font-size:10px;color:rgba(255,255,255,0.4)">{{ currentUser?.rol }}</div>
            </div>
            <v-chip
              :color="currentUser?.rol === 'ADMINISTRADOR' ? 'primary' : 'secondary'"
              size="x-small"
              variant="tonal"
            >
              {{ currentUser?.rol === 'ADMINISTRADOR' ? 'ADMIN' : 'USR' }}
            </v-chip>
          </div>
        </div>
      </Transition>

      <!-- Nav items -->
      <v-list density="compact" nav class="px-2">

        <!-- General -->
        <div v-if="!rail" class="nav-section-label">GENERAL</div>
        <v-list-item
          v-for="item in sharedNav"
          :key="item.to"
          :to="item.to"
          :prepend-icon="item.icon"
          :title="item.label"
          :value="item.to"
          rounded="lg"
          active-color="primary"
          class="nav-item mb-1"
        />

        <!-- Admin -->
        <template v-if="isAdmin">
          <v-divider class="border-opacity-10 my-2" />
          <div v-if="!rail" class="nav-section-label">ADMINISTRACIÓN</div>
          <v-list-item
            v-for="item in adminNav"
            :key="item.to"
            :to="item.to"
            :prepend-icon="item.icon"
            :title="item.label"
            :value="item.to"
            rounded="lg"
            active-color="primary"
            class="nav-item mb-1"
          />
        </template>

        <!-- Cliente -->
        <template v-if="!isAdmin">
          <v-divider class="border-opacity-10 my-2" />
          <div v-if="!rail" class="nav-section-label">MI CUENTA</div>
          <v-list-item
            v-for="item in clienteNav"
            :key="item.to"
            :to="item.to"
            :prepend-icon="item.icon"
            :title="item.label"
            :value="item.to"
            rounded="lg"
            active-color="secondary"
            class="nav-item mb-1"
          />
        </template>

      </v-list>

      <template #append>
        <v-divider class="border-opacity-10 mb-2" />
        <div class="px-2 pb-3">
          <v-list-item
            to="/perfil"
            prepend-icon="mdi-account-circle-outline"
            title="Mi perfil"
            rounded="lg"
            class="nav-item mb-1"
          />
          <v-list-item
            prepend-icon="mdi-logout"
            title="Cerrar sesión"
            rounded="lg"
            class="nav-item text-error"
            @click="handleLogout"
          />
        </div>
      </template>
    </v-navigation-drawer>

    <!-- ── Main ────────────────────────────────────────────── -->
    <v-main class="main-bg">
      <!-- Topbar -->
      <div class="topbar">
        <div class="topbar-left">
          <h2 class="page-title">{{ pageTitle }}</h2>
          <div class="breadcrumb mono">{{ pageBreadcrumb }}</div>
        </div>
        <div class="topbar-right">
          <v-chip color="success" variant="tonal" size="small" class="mono mr-2">
            <v-icon start size="10">mdi-circle</v-icon>
            API Connected
          </v-chip>
          <v-btn icon="mdi-bell-outline" variant="text" size="small" />
        </div>
      </div>

      <!-- Page content -->
      <div class="page-content">
        <router-view v-slot="{ Component }">
          <Transition name="page" mode="out-in">
            <component :is="Component" />
          </Transition>
        </router-view>
      </div>
    </v-main>

  </v-layout>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuth } from '@/composables/useAuth'

const { currentUser, isAdmin, logout } = useAuth()
const router = useRouter()
const route  = useRoute()
const drawer = ref(true)
const rail   = ref(false)

const userInitial = computed(() => currentUser.value?.nombre?.charAt(0).toUpperCase() || 'U')

const sharedNav = [
  { to: '/dashboard', icon: 'mdi-view-dashboard-outline', label: 'Dashboard' },
]

const adminNav = [
  { to: '/usuarios',       icon: 'mdi-account-group-outline',  label: 'Usuarios' },
  { to: '/equipos',        icon: 'mdi-desktop-classic',         label: 'Equipos' },
  { to: '/prestamos',      icon: 'mdi-hand-extended-outline',   label: 'Préstamos' },
  { to: '/penalizaciones', icon: 'mdi-alert-circle-outline',    label: 'Penalizaciones' },
]

const clienteNav = [
  { to: '/catalogo',           icon: 'mdi-view-grid-outline',     label: 'Catálogo de equipos' },
  { to: '/mis-prestamos',      icon: 'mdi-hand-extended-outline', label: 'Mis préstamos' },
  { to: '/mis-penalizaciones', icon: 'mdi-shield-alert-outline',  label: 'Mis penalizaciones' },
]

const pageMeta = {
  '/dashboard':          { title: 'Dashboard',           breadcrumb: 'sgplab / inicio' },
  '/usuarios':           { title: 'Usuarios',            breadcrumb: 'sgplab / admin / usuarios' },
  '/equipos':            { title: 'Equipos',             breadcrumb: 'sgplab / admin / equipos' },
  '/prestamos':          { title: 'Préstamos',           breadcrumb: 'sgplab / admin / préstamos' },
  '/penalizaciones':     { title: 'Penalizaciones',      breadcrumb: 'sgplab / admin / penalizaciones' },
  '/catalogo':           { title: 'Catálogo de equipos', breadcrumb: 'sgplab / equipos disponibles' },
  '/mis-prestamos':      { title: 'Mis préstamos',       breadcrumb: 'sgplab / mi cuenta / préstamos' },
  '/mis-penalizaciones': { title: 'Mis penalizaciones',  breadcrumb: 'sgplab / mi cuenta / penalizaciones' },
  '/perfil':             { title: 'Mi perfil',           breadcrumb: 'sgplab / perfil' },
}

const pageTitle      = computed(() => pageMeta[route.path]?.title || 'SGP Lab')
const pageBreadcrumb = computed(() => pageMeta[route.path]?.breadcrumb || '')

function handleLogout() {
  logout()
  router.push('/login')
}
</script>

<style scoped>
.app-layout {
  min-height: 100vh;
  background: #0a0e14;
}

.sidebar {
  border-right: 1px solid rgba(255,255,255,0.06) !important;
}

.sidebar-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 16px 12px;
  cursor: pointer;
  user-select: none;
  min-height: 64px;
}

.brand-dot {
  width: 36px; height: 36px;
  background: rgba(0,212,255,0.1);
  border: 1px solid rgba(0,212,255,0.2);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}

.brand-name {
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: 3px;
  color: #e2e8f0;
  white-space: nowrap;
}

.user-chip {
  display: flex;
  align-items: center;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px;
  padding: 8px 10px;
}

.nav-section-label {
  font-size: 0.6rem;
  letter-spacing: 2px;
  color: rgba(255,255,255,0.25);
  font-weight: 600;
  padding: 4px 12px 6px;
  font-family: 'Space Mono', monospace;
  white-space: nowrap;
}

.nav-item {
  font-size: 0.875rem !important;
}

/* Main area */
.main-bg {
  background: #0a0e14 !important;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 28px 16px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  background: rgba(17,24,32,0.8);
  backdrop-filter: blur(8px);
  position: sticky;
  top: 0;
  z-index: 10;
}

.page-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #e2e8f0;
}

.breadcrumb {
  font-size: 0.7rem;
  color: rgba(255,255,255,0.3);
  margin-top: 2px;
}

.topbar-right {
  display: flex;
  align-items: center;
}

.page-content {
  padding: 28px;
}

/* Transitions */
.fade-enter-active, .fade-leave-active { transition: opacity 0.15s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

.page-enter-active { transition: all 0.2s ease; }
.page-leave-active { transition: all 0.15s ease; }
.page-enter-from { opacity: 0; transform: translateY(8px); }
.page-leave-to   { opacity: 0; transform: translateY(-4px); }
</style>
