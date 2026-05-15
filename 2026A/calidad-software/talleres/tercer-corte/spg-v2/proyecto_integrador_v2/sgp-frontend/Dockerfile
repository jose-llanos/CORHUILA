# =====================================================================
# SGP LAB Frontend - Dockerfile multi-stage
#
# Stage 1 (builder): instala dependencias npm y compila el bundle Vite.
# Stage 2 (runtime): Nginx Alpine sirviendo los estaticos.
#
# Build:
#   docker build -t sgp-lab/frontend:1.0.0 ./sgp-frontend
#
# Run:
#   docker run --rm -p 5173:80 sgp-lab/frontend:1.0.0
# =====================================================================

# ---------- STAGE 1: build ----------
FROM node:20-alpine AS builder

WORKDIR /build

# Copiar manifests primero para cachear node_modules
COPY package*.json ./
RUN npm ci --no-audit --no-fund

# Copiar el resto del codigo y construir
COPY . .

# Permite parametrizar la URL del backend en build-time si quieres
ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN VITE_API_BASE_URL=/api npm run build

# ---------- STAGE 2: runtime ----------
FROM nginx:1.27-alpine AS runtime

# Quitar la config default y poner la nuestra (SPA fallback + proxy /api)
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copiar el bundle estatico al directorio que Nginx sirve
COPY --from=builder /build/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --quiet --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
