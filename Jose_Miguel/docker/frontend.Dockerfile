# Dockerfile para Angular 19 - AutoSpark Frontend
FROM node:20-alpine AS build
WORKDIR /app

# Copiar archivos de configuración
COPY app/frontend/package*.json ./

# Instalar dependencias
RUN npm ci --force || npm install --force

# Copiar código fuente
COPY app/frontend/ ./

# Construir la aplicación para producción (Angular 19)
RUN npm run build -- --configuration=production

# Servir con Nginx
FROM nginx:alpine

# Configuración de Nginx para SPA y proxy al backend
RUN echo 'server { \
    listen 80; \
    server_name localhost; \
    root /usr/share/nginx/html; \
    index index.html; \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
    location /api/ { \
        proxy_pass http://backend:8080/api/; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
    } \
}' > /etc/nginx/conf.d/default.conf

# Copiar los archivos compilados (Angular 19 genera en dist/autospark-frontend/)
COPY --from=build /app/dist/autospark_frontend/ /usr/share/nginx/html/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
