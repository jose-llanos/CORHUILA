-- Crear base de datos si no existe
CREATE DATABASE IF NOT EXISTS autospark;
USE autospark;

-- Crear usuario si no existe
CREATE USER IF NOT EXISTS 'autospark_user'@'%' IDENTIFIED BY 'autospark_password';

-- Dar permisos
GRANT ALL PRIVILEGES ON autospark.* TO 'autospark_user'@'%';
FLUSH PRIVILEGES;
