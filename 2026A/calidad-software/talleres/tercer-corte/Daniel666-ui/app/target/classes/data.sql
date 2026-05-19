-- Datos iniciales para la base H2.
--
-- Nota: se utiliza {noop} para que Spring Security acepte passwords sin hashing en el seed,
-- manteniendo credenciales simples para pruebas (admin/admin, user/user).

INSERT INTO users (username, email, password, role) VALUES
('admin', 'admin@example.com', '{noop}admin', 'ADMIN'),
('user', 'user@example.com', '{noop}user', 'USER');

INSERT INTO tasks (title, description, status, priority, created_at, updated_at) VALUES
('Prepare project structure', 'Create base folders and Maven module', 'PENDING', 'HIGH', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
('Implement Task CRUD', 'Create controllers and API endpoints for tasks', 'IN_PROGRESS', 'HIGH', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
('Add QA assets', 'Add tests and reports generation', 'PENDING', 'MEDIUM', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),
('Finalize documentation', 'Write formal documentation in Spanish', 'PENDING', 'LOW', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP());
