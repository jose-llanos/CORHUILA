// useAuth.js - estado de autenticacion global con JWT
import { ref, computed } from 'vue'

const STORAGE_KEY = 'sgp_session_v2'

// Estado reactivo global (singleton compartido por toda la app)
const session = ref(loadFromStorage())

function loadFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    // Validar expiracion
    if (parsed.expiresAt && Date.now() > parsed.expiresAt) {
      localStorage.removeItem(STORAGE_KEY)
      return null
    }
    return parsed
  } catch {
    return null
  }
}

function saveToStorage(value) {
  if (value) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(value))
  } else {
    localStorage.removeItem(STORAGE_KEY)
  }
}

export function useAuth() {
  const isAuthenticated = computed(() => session.value !== null && session.value.token)
  const role = computed(() => session.value?.rol || null)
  const isAdmin = computed(() => role.value === 'ADMINISTRADOR')
  const user = computed(() => session.value ? {
    id: session.value.userId,
    email: session.value.email,
    nombre: session.value.nombre,
    rol: session.value.rol,
    estado: 'ACTIVO'
  } : null)
  // Alias retro-compatible con views existentes
  const currentUser = user
  const token = computed(() => session.value?.token || null)

  /**
   * Establece la sesion tras un login exitoso.
   * @param {Object} loginResponse - respuesta de /api/auth/login
   */
  function setSession(loginResponse) {
    const expiresAt = Date.now() + (loginResponse.expiresInMs || 3600000)
    const value = {
      token: loginResponse.token,
      userId: loginResponse.userId,
      email: loginResponse.email,
      nombre: loginResponse.nombre,
      rol: loginResponse.rol,
      expiresAt
    }
    session.value = value
    saveToStorage(value)
  }

  function logout() {
    session.value = null
    saveToStorage(null)
  }

  /**
   * Limpia la sesion silenciosamente (usado por el interceptor en 401).
   */
  function clearSession() {
    session.value = null
    saveToStorage(null)
  }

  function isExpired() {
    if (!session.value?.expiresAt) return false
    return Date.now() > session.value.expiresAt
  }

  /**
   * Stub legacy: algunos views existentes invocan login({...currentUser, nombre, email})
   * tras actualizar el perfil. Aqui solo refrescamos los campos locales del session.
   * No emite peticion al backend ni reemite token.
   */
  function login(partialUser) {
    if (!session.value || !partialUser) return
    const updated = {
      ...session.value,
      nombre: partialUser.nombre ?? session.value.nombre,
      email: partialUser.email ?? session.value.email
    }
    session.value = updated
    saveToStorage(updated)
  }

  return {
    isAuthenticated,
    role,
    isAdmin,
    user,
    currentUser,
    token,
    setSession,
    login,
    logout,
    clearSession,
    isExpired
  }
}
