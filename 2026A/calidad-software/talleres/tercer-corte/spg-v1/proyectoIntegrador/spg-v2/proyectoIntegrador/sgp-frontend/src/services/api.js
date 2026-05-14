import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' }
})

// ─── Usuarios ───────────────────────────────────────────
export const usuarioService = {
  getAll: () => api.get('/usuarios'),
  getById: (id) => api.get(`/usuarios/${id}`),
  create: (data) => api.post('/usuarios', data),
  update: (id, data) => api.put(`/usuarios/${id}`, data),
  delete: (id) => api.delete(`/usuarios/${id}`)
}

// ─── Equipos ────────────────────────────────────────────
export const equipoService = {
  getAll: () => api.get('/equipos'),
  getById: (id) => api.get(`/equipos/${id}`),
  create: (data) => api.post('/equipos', data),
  update: (id, data) => api.put(`/equipos/${id}`, data),
  delete: (id) => api.delete(`/equipos/${id}`)
}

// ─── Préstamos ──────────────────────────────────────────
export const prestamoService = {
  getAll: () => api.get('/prestamos'),
  getById: (id) => api.get(`/prestamos/${id}`),
  create: (data) => api.post('/prestamos', data),
  update: (id, data) => api.put(`/prestamos/${id}`, data),
  delete: (id) => api.delete(`/prestamos/${id}`)
}

// ─── Penalizaciones ─────────────────────────────────────
export const penalizacionService = {
  getAll: () => api.get('/penalizaciones'),
  getById: (id) => api.get(`/penalizaciones/${id}`),
  checkUsuario: (usuarioId) => api.get(`/penalizaciones/usuario/${usuarioId}/activa`),
  create: (data) => api.post('/penalizaciones', data),
  update: (id, data) => api.put(`/penalizaciones/${id}`, data),
  delete: (id) => api.delete(`/penalizaciones/${id}`)
}

export default api
