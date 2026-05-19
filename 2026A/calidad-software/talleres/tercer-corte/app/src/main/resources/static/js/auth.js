document.addEventListener('DOMContentLoaded', () => {
    const loginCard = document.getElementById('login-card');
    const registerCard = document.getElementById('register-card');
    const authError = document.getElementById('auth-error');

    // Alternancia de vistas
    document.getElementById('go-to-register').addEventListener('click', (e) => {
        e.preventDefault();
        loginCard.classList.add('hidden');
        registerCard.classList.remove('hidden');
        authError.classList.add('hidden');
    });

    document.getElementById('go-to-login').addEventListener('click', (e) => {
        e.preventDefault();
        registerCard.classList.add('hidden');
        loginCard.classList.remove('hidden');
        authError.classList.add('hidden');
    });

    // Formulario de Login
    document.getElementById('login-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('login-username').value;
        const password = document.getElementById('login-password').value;

        try {
            const res = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username: username, password: password })
            });
            
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.message || 'Credenciales inválidas');
            }

            localStorage.setItem('token', data.token);
            window.location.href = '/dashboard.html';
        } catch (err) {
            authError.textContent = err.message;
            authError.classList.remove('hidden');
        }
    });

    // Formulario de Registro
    document.getElementById('register-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('reg-username').value;
        const email = document.getElementById('reg-email').value;
        const password = document.getElementById('reg-password').value;

        try {
            const res = await fetch('/api/auth/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username: username, email: email, password: password })
            });
            
            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.message || 'Error en el registro');
            }
            
            alert('Cuenta creada con éxito. Inicia sesión.');
            document.getElementById('go-to-login').click();
        } catch (err) {
            authError.textContent = err.message;
            authError.classList.remove('hidden');
        }
    });
});