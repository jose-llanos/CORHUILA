// api.js - cliente axios con interceptor JWT
import axios from 'axios'
import { useAuth } from '@/composables/useAuth'

const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080/api'

const api = axios.create({
  baseURL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 15000
})

// Inyectar token Bearer en cada request
api.interceptors.request.use(
  (config) => {
    const { token } = useAuth()
    if (token.value) {
      config.headers.Authorization = `Bearer ${token.value}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Manejar respuestas: en 401 limpiar sesion y redirigir
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error.response?.status
    if (status === 401) {
      const { clearSession } = useAuth()
      clearSession()
      if (!window.location.pathname.includes('/login')) {
        window.location.href = '/login?expired=1'
      }
    }
    return Promise.reject(error)
  }
)

// ============ ENDPOINTS DE AUTENTICACION ============
export const authApi = {
  login: (email, password) =>
    api.post('/auth/login', { email, password }).then(r => r.data)
}

// ============ ENDPOINTS DE USUARIOS ============
export const usuariosApi = {
  listar: () => api.get('/usuarios').then(r => r.data),
  obtener: (id) => api.get(`/usuarios/${id}`).then(r => r.data),
  crear: (data) => api.post('/usuarios', data).then(r => r.data),
  actualizar: (id, data) => api.put(`/usuarios/${id}`, data).then(r => r.data),
  eliminar: (id) => api.delete(`/usuarios/${id}`).then(r => r.data)
}

// ============ ENDPOINTS DE EQUIPOS ============
export const equiposApi = {
  listar: () => api.get('/equipos').then(r => r.data),
  obtener: (id) => api.get(`/equipos/${id}`).then(r => r.data),
  crear: (data) => api.post('/equipos', data).then(r => r.data),
  actualizar: (id, data) => api.put(`/equipos/${id}`, data).then(r => r.data),
  eliminar: (id) => api.delete(`/equipos/${id}`).then(r => r.data)
}

// ============ ENDPOINTS DE PRESTAMOS ============
export const prestamosApi = {
  listar: () => api.get('/prestamos').then(r => r.data),
  obtener: (id) => api.get(`/prestamos/${id}`).then(r => r.data),
  crear: (data) => api.post('/prestamos', data).then(r => r.data),
  actualizar: (id, data) => api.put(`/prestamos/${id}`, data).then(r => r.data),
  eliminar: (id) => api.delete(`/prestamos/${id}`).then(r => r.data)
}

// ============ ENDPOINTS DE PENALIZACIONES ============
export const penalizacionesApi = {
  listar: () => api.get('/penalizaciones').then(r => r.data),
  obtener: (id) => api.get(`/penalizaciones/${id}`).then(r => r.data),
  crear: (data) => api.post('/penalizaciones', data).then(r => r.data),
  actualizar: (id, data) => api.put(`/penalizaciones/${id}`, data).then(r => r.data),
  eliminar: (id) => api.delete(`/penalizaciones/${id}`).then(r => r.data),
  verificarActiva: (usuarioId) => api.get(`/penalizaciones/usuario/${usuarioId}/activa`).then(r => r.data)
}

/**
 * Extrae un mensaje de error legible desde un error de axios.
 * @param {Error} error - error capturado en un catch
 * @returns {string}
 */
export function getErrorMessage(error) {
  const data = error?.response?.data
  if (data?.message) {
    if (data.fieldErrors?.length) {
      const fieldMsgs = data.fieldErrors.map(fe => `${fe.field}: ${fe.message}`).join('; ')
      return `${data.message} (${fieldMsgs})`
    }
    return data.message
  }
  if (error?.message) return error.message
  return 'Error desconocido'
}

export default api

// ============ WRAPPERS RETRO-COMPATIBLES (estilo axios: response.data) ============
// Los views existentes esperan res.data, mantenemos esa interfaz tambien.

function withData(promise) {
  return promise.then(data => ({ data }))
}

/**
 * Adaptador para prestamos/penalizaciones:
 * el backend devuelve campos planos (usuarioId, usuarioNombre, equipoId, equipoNombre)
 * pero los views existentes consumen objetos anidados (usuario.id, equipo.nombre, etc).
 * Esta funcion reconstruye los objetos anidados para mantener compatibilidad.
 */
function adaptPrestamoResponse(p) {
  if (!p) return p
  return {
    ...p,
    usuario: p.usuario || (p.usuarioId != null ? { id: p.usuarioId, nombre: p.usuarioNombre } : null),
    equipo: p.equipo || (p.equipoId != null ? { id: p.equipoId, nombre: p.equipoNombre } : null)
  }
}
function adaptListPrestamo(list) {
  return Array.isArray(list) ? list.map(adaptPrestamoResponse) : list
}
function adaptPenalizacionResponse(p) {
  if (!p) return p
  return {
    ...p,
    usuario: p.usuario || (p.usuarioId != null ? { id: p.usuarioId, nombre: p.usuarioNombre } : null)
  }
}
function adaptListPenalizacion(list) {
  return Array.isArray(list) ? list.map(adaptPenalizacionResponse) : list
}

export const usuarioService = {
  list:    () => withData(usuariosApi.listar()),
  getAll:  () => withData(usuariosApi.listar()),
  get:     (id) => withData(usuariosApi.obtener(id)),
  create:  (data) => withData(usuariosApi.crear(data)),
  update:  (id, data) => withData(usuariosApi.actualizar(id, data)),
  remove:  (id) => withData(usuariosApi.eliminar(id)),
  delete:  (id) => withData(usuariosApi.eliminar(id))
}
export const equipoService = {
  list:    () => withData(equiposApi.listar()),
  getAll:  () => withData(equiposApi.listar()),
  get:     (id) => withData(equiposApi.obtener(id)),
  create:  (data) => withData(equiposApi.crear(data)),
  update:  (id, data) => withData(equiposApi.actualizar(id, data)),
  remove:  (id) => withData(equiposApi.eliminar(id)),
  delete:  (id) => withData(equiposApi.eliminar(id))
}
export const prestamoService = {
  list:    () => prestamosApi.listar().then(d => ({ data: adaptListPrestamo(d) })),
  getAll:  () => prestamosApi.listar().then(d => ({ data: adaptListPrestamo(d) })),
  get:     (id) => prestamosApi.obtener(id).then(d => ({ data: adaptPrestamoResponse(d) })),
  create:  (data) => prestamosApi.crear(data).then(d => ({ data: adaptPrestamoResponse(d) })),
  update:  (id, data) => prestamosApi.actualizar(id, data).then(d => ({ data: adaptPrestamoResponse(d) })),
  remove:  (id) => withData(prestamosApi.eliminar(id)),
  delete:  (id) => withData(prestamosApi.eliminar(id))
}
export const penalizacionService = {
  list:    () => penalizacionesApi.listar().then(d => ({ data: adaptListPenalizacion(d) })),
  getAll:  () => penalizacionesApi.listar().then(d => ({ data: adaptListPenalizacion(d) })),
  get:     (id) => penalizacionesApi.obtener(id).then(d => ({ data: adaptPenalizacionResponse(d) })),
  create:  (data) => penalizacionesApi.crear(data).then(d => ({ data: adaptPenalizacionResponse(d) })),
  update:  (id, data) => penalizacionesApi.actualizar(id, data).then(d => ({ data: adaptPenalizacionResponse(d) })),
  remove:  (id) => withData(penalizacionesApi.eliminar(id)),
  delete:  (id) => withData(penalizacionesApi.eliminar(id)),
  checkUsuario: (usuarioId) => withData(penalizacionesApi.verificarActiva(usuarioId))
}
