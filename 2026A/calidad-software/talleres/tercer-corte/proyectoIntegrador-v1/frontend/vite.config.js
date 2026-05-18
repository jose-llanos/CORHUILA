import { defineConfig } from 'vite';
import { resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  root: '.',
  publicDir: false,
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        authLogin: resolve(__dirname, 'pages/auth/login.html'),
        authRegister: resolve(__dirname, 'pages/auth/register.html'),
        adminDashboard: resolve(__dirname, 'pages/admin/dashboard.html'),
        adminAppointments: resolve(__dirname, 'pages/admin/appointments.html'),
        adminDoctors: resolve(__dirname, 'pages/admin/doctors.html'),
        adminLeaves: resolve(__dirname, 'pages/admin/leaves.html'),
        adminSpecialties: resolve(__dirname, 'pages/admin/specialties.html'),
        adminSchedules: resolve(__dirname, 'pages/admin/schedules.html'),
        doctorDashboard: resolve(__dirname, 'pages/doctor/dashboard.html'),
        doctorAppointments: resolve(__dirname, 'pages/doctor/appointments.html'),
        doctorSchedule: resolve(__dirname, 'pages/doctor/schedule.html'),
        doctorLeaves: resolve(__dirname, 'pages/doctor/leaves.html'),
        patientDashboard: resolve(__dirname, 'pages/patient/dashboard.html'),
        patientAppointments: resolve(__dirname, 'pages/patient/appointments.html'),
        patientHistory: resolve(__dirname, 'pages/patient/history.html')
      }
    }
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true
      }
    }
  }
});
