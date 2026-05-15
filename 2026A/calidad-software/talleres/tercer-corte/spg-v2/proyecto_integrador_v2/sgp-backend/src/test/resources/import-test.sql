-- Datos de carga para perfil TEST (H2 con modo PostgreSQL)
INSERT INTO usuarios (nombre, email, password_hash, estado, rol) VALUES
('Administrador SGP', 'admin@sgplab.edu.co',
 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6:6caa313cc1746666ed773360d66917beeea2fcf8856c299f16d047440ac463d2',
 'ACTIVO', 'ADMINISTRADOR');

INSERT INTO usuarios (nombre, email, password_hash, estado, rol) VALUES
('Cliente Demo', 'cliente@sgplab.edu.co',
 '1234567890abcdef1122334455667788:d928a558fb55b3b7345b9535b43f81dda3eb32ef8d1e953a183c7d3212949e40',
 'ACTIVO', 'CLIENTE');

INSERT INTO equipos (nombre, cantidad, estado, codigo_inventario) VALUES
('Microscopio Optico', 5, 'DISPONIBLE', 'MIC-001'),
('Centrifuga Digital', 2, 'DISPONIBLE', 'CEN-002'),
('Espectrofotometro UV', 1, 'DISPONIBLE', 'ESP-003');
