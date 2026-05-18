# Stage 1: Build
FROM node:20-alpine AS build
WORKDIR /build

COPY package.json ./
RUN npm install

COPY vite.config.js ./
COPY index.html ./
COPY css ./css
COPY js ./js
COPY pages ./pages
RUN npm run build

# Stage 2: Runtime
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /build/dist /usr/share/nginx/html
EXPOSE 80
