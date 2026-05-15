-- ============================================================
-- DATOS INICIALES PARA EL PERFIL DEV (PostgreSQL)
-- ============================================================
-- Hashes generados con SHA-256(salt || password) en hex:
--   admin@sgplab.edu.co  / password    -> hash de 'password'
--   cliente@sgplab.edu.co / cliente123 -> hash de 'cliente123'
-- ============================================================

INSERT INTO usuarios (nombre, email, password_hash, estado, rol)
SELECT 'Administrador SGP', 'admin@sgplab.edu.co',
       'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6:6caa313cc1746666ed773360d66917beeea2fcf8856c299f16d047440ac463d2',
       'ACTIVO', 'ADMINISTRADOR'
WHERE NOT EXISTS (SELECT 1 FROM usuarios WHERE email = 'admin@sgplab.edu.co');

INSERT INTO usuarios (nombre, email, password_hash, estado, rol)
SELECT 'Cliente Demo', 'cliente@sgplab.edu.co',
       '1234567890abcdef1122334455667788:d928a558fb55b3b7345b9535b43f81dda3eb32ef8d1e953a183c7d3212949e40',
       'ACTIVO', 'CLIENTE'
WHERE NOT EXISTS (SELECT 1 FROM usuarios WHERE email = 'cliente@sgplab.edu.co');

INSERT INTO equipos (nombre, cantidad, estado, codigo_inventario)
SELECT 'Microscopio Optico', 5, 'DISPONIBLE', 'MIC-001'
WHERE NOT EXISTS (SELECT 1 FROM equipos WHERE codigo_inventario = 'MIC-001');

INSERT INTO equipos (nombre, cantidad, estado, codigo_inventario)
SELECT 'Centrifuga Digital', 2, 'DISPONIBLE', 'CEN-002'
WHERE NOT EXISTS (SELECT 1 FROM equipos WHERE codigo_inventario = 'CEN-002');

INSERT INTO equipos (nombre, cantidad, estado, codigo_inventario)
SELECT 'Espectrofotometro UV', 1, 'DISPONIBLE', 'ESP-003'
WHERE NOT EXISTS (SELECT 1 FROM equipos WHERE codigo_inventario = 'ESP-003');
