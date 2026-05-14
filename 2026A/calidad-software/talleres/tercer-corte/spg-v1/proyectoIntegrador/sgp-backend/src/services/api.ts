import axios from 'axios';

// Configuramos la URL base de tu Spring Boot (por defecto suele ser 8080)
export const api = axios.create({
  baseURL: 'http://localhost:8080/api',
  headers: {
    'Content-Type': 'application/json'
  }
});