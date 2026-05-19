const API_BASE = '/api';

const client = {
    async request(url, options = {}) {
        const token = localStorage.getItem('token'); 
        options.headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };

        if (token) {
            options.headers['Authorization'] = `Bearer ${token}`; 
        }

        const response = await fetch(`${API_BASE}${url}`, options);

        if (response.status === 401 || response.status === 403) {
            localStorage.removeItem('token'); 
            window.location.href = '/index.html'; 
            return;
        }

        if (response.status === 204) return null;

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.message || 'Error en la operación del servidor.');
        }
        return data;
    },
    get(url) { return this.request(url, { method: 'GET' }); },
    post(url, body) { return this.request(url, { method: 'POST', body: JSON.stringify(body) }); },
    put(url, body) { return this.request(url, { method: 'PUT', body: JSON.stringify(body) }); },
    patch(url, body) { return this.request(url, { method: 'PATCH', body: JSON.stringify(body) }); },
    delete(url) { return this.request(url, { method: 'DELETE' }); }
};