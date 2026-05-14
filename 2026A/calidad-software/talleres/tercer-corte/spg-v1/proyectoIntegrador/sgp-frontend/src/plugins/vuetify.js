import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import 'vuetify/styles'

const sgpTheme = {
  dark: true,
  colors: {
    background: '#0a0e14',
    surface: '#111820',
    'surface-variant': '#1a2332',
    primary: '#00d4ff',
    'primary-darken-1': '#0099bb',
    secondary: '#ff6b35',
    'secondary-darken-1': '#cc5522',
    accent: '#7c3aed',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#3b82f6',
    'on-background': '#e2e8f0',
    'on-surface': '#e2e8f0',
    'on-primary': '#000000',
  }
}

export default createVuetify({
  components,
  directives,
  theme: {
    defaultTheme: 'sgpTheme',
    themes: { sgpTheme }
  },
  defaults: {
    VBtn: { rounded: 'lg', fontWeight: '500' },
    VCard: { rounded: 'xl', elevation: 0 },
    VTextField: { variant: 'outlined', density: 'comfortable' },
    VSelect: { variant: 'outlined', density: 'comfortable' },
    VDataTable: { density: 'comfortable' },
  }
})
