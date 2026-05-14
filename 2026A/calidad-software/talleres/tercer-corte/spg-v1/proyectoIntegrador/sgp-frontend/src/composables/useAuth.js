import { ref, computed } from 'vue'

const currentUser = ref(JSON.parse(localStorage.getItem('sgp_user') || 'null'))

export function useAuth() {
  const isAuthenticated = computed(() => currentUser.value !== null)
  const isAdmin = computed(() => currentUser.value?.rol === 'ADMINISTRADOR')
  const isCliente = computed(() => currentUser.value?.rol === 'CLIENTE')

  function login(userData) {
    currentUser.value = userData
    localStorage.setItem('sgp_user', JSON.stringify(userData))
  }

  function logout() {
    currentUser.value = null
    localStorage.removeItem('sgp_user')
  }

  return {
    currentUser,
    isAuthenticated,
    isAdmin,
    isCliente,
    login,
    logout
  }
}
