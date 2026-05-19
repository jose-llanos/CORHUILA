let currentUser = null;
let currentProject = null;
let projectMembers = [];

document.addEventListener('DOMContentLoaded', async () => {
    if (!localStorage.getItem('token')) {
        window.location.href = '/index.html';
        return;
    }
    await initDashboard();
});

async function initDashboard() {
    try {
        currentUser = await client.get('/auth/me');
        document.getElementById('user-display').textContent = currentUser.username;
        setupGlobalEvents();
        await loadProjectsList();
    } catch (e) {
        logout();
    }
}

function setupGlobalEvents() {
    document.getElementById('btn-logout').addEventListener('click', logout);
    document.getElementById('open-project-modal').addEventListener('click', openCreateProjectModal);
    document.getElementById('open-task-modal').addEventListener('click', openCreateTaskModal);
    document.getElementById('close-modal-btn').addEventListener('click', closeModal);
    document.getElementById('user-display').addEventListener('click', openProfileModal);
}

function logout() {
    localStorage.removeItem('token');
    window.location.href = '/index.html';
}

async function loadProjectsList() {
    const listContainer = document.getElementById('projects-list');
    listContainer.innerHTML = '';
    const projects = await client.get('/projects');

    projects.forEach(p => {
        const li = document.createElement('li');
        li.textContent = p.name;
        li.setAttribute('data-test', `project-item-${p.id}`);
        li.addEventListener('click', () => selectProject(p.id));
        listContainer.appendChild(li);
    });
}

async function selectProject(id) {
    document.getElementById('empty-state').classList.add('hidden');
    const detailView = document.getElementById('project-detail');
    detailView.classList.remove('hidden');

    currentProject = await client.get(`/projects/${id}`);
    projectMembers = await client.get(`/projects/${id}/members`);

    renderProjectHeader();
    renderMembersSection();
    renderKanbanBoard();
}

function renderProjectHeader() {
    document.getElementById('view-project-name').textContent = currentProject.name;
    document.getElementById('view-project-desc').textContent = currentProject.description || '';
    
    const actionsBox = document.getElementById('owner-actions-container');
    actionsBox.innerHTML = '';

    if (currentProject.owner.id === currentUser.id) {
        actionsBox.innerHTML = `
            <button onclick="openEditProjectModal()" data-test="btn-edit-project">Editar</button>
            <button onclick="deleteProject()" class="btn-danger" data-test="btn-delete-project">Eliminar</button>
        `;
    }
}

function renderMembersSection() {
    const wrapper = document.getElementById('invite-member-wrapper');
    wrapper.innerHTML = '';

    if (currentProject.owner.id === currentUser.id) {
        wrapper.innerHTML = `
            <input type="text" id="invite-username" placeholder="Username de invitado" data-test="input-invite-username">
            <button onclick="executeInvitation()" data-test="btn-submit-invite">Invitar</button>
        `;
    }

    const listUl = document.getElementById('project-members-list');
    listUl.innerHTML = '';

    projectMembers.forEach(m => {
        const li = document.createElement('li');
        li.setAttribute('data-test', `member-${m.id}`);
        li.innerHTML = `
            <span>${m.username} ${m.owner ? '<strong>(Owner)</strong>' : ''}</span>
        `;
        
        if (currentProject.owner.id === currentUser.id && !m.owner) {
            li.innerHTML += `<button class="btn-sm btn-danger" onclick="removeMember(${m.id})" data-test="btn-remove-member-${m.id}">Remover</button>`;
        }
        listUl.appendChild(li);
    });
}

function renderKanbanBoard() {
    const isOwner = currentProject.owner.id === currentUser.id;
    const columns = {
        'PENDING': document.getElementById('tasks-pending'),
        'IN_PROGRESS': document.getElementById('tasks-in-progress'),
        'DONE': document.getElementById('tasks-done')
    };

    Object.values(columns).forEach(c => c.innerHTML = '');

    const groups = ['pending', 'inProgress', 'done'];
    groups.forEach(groupKey => {
        if (!currentProject[groupKey]) return;
        
        currentProject[groupKey].forEach(task => {
            const card = document.createElement('div');
            card.className = 'task-card';
            card.setAttribute('data-test', `task-card-${task.id}`);

            const assignedName = task.assignedTo ? task.assignedTo.username : 'Sin asignar';
            const canChangeStatus = isOwner || (task.assignedTo && task.assignedTo.id === currentUser.id);

            let statusSelectHtml = '';
            if (canChangeStatus) {
                statusSelectHtml = `
                    <select onchange="changeTaskStatus(${task.id}, this.value)" data-test="select-status-${task.id}">
                        <option value="PENDING" ${task.status === 'PENDING' ? 'selected' : ''}>Pendiente</option>
                        <option value="IN_PROGRESS" ${task.status === 'IN_PROGRESS' ? 'selected' : ''}>En Progreso</option>
                        <option value="DONE" ${task.status === 'DONE' ? 'selected' : ''}>Terminada</option>
                    </select>
                `;
            }

            let memberOptions = `<option value="">Desasignar</option>`;
            projectMembers.forEach(m => {
                memberOptions += `<option value="${m.id}" ${task.assignedTo && task.assignedTo.id === m.id ? 'selected' : ''}>${m.username}</option>`;
            });

            let assignmentHtml = `<span>Asignado a: <strong>${assignedName}</strong></span>`;
            if (isOwner) {
                assignmentHtml = `
                    <label>Asignar:</label>
                    <select onchange="assignTask(${task.id}, this.value)" data-test="select-assign-${task.id}">
                        ${memberOptions}
                    </select>
                `;
            }

            let managementButtons = '';
            if (isOwner) {
                managementButtons = `<button class="btn-sm btn-danger" onclick="deleteTask(${task.id})" data-test="btn-delete-task-${task.id}">Eliminar</button>`;
            }

            card.innerHTML = `
                <h5>${task.title}</h5>
                <p>${task.description || ''}</p>
                <div class="task-meta">
                    ${assignmentHtml}
                    ${statusSelectHtml}
                </div>
                <div class="task-actions">
                    <button class="btn-sm" onclick="openEditTaskModal(${task.id}, '${task.title}', '${task.description || ''}')" data-test="btn-edit-task-${task.id}">Editar</button>
                    ${managementButtons}
                </div>
            `;

            const containerId = task.status === 'PENDING' ? 'PENDING' : (task.status === 'IN_PROGRESS' ? 'IN_PROGRESS' : 'DONE');
            if (columns[containerId]) {
                columns[containerId].appendChild(card);
            }
        });
    });
}

// OPERACIONES CRUD & API BINDING
async function executeInvitation() {
    const usernameInput = document.getElementById('invite-username');
    const username = usernameInput.value;
    if(!username) return;
    try {
        await client.post(`/projects/${currentProject.id}/members`, { username });
        usernameInput.value = '';
        await selectProject(currentProject.id);
    } catch(e) { alert(e.message); }
}

async function removeMember(userId) {
    if(confirm("¿Remover miembro?")) {
        await client.delete(`/projects/${currentProject.id}/members/${userId}`);
        await selectProject(currentProject.id);
    }
}

async function changeTaskStatus(taskId, newStatus) {
    await client.patch(`/projects/${currentProject.id}/tasks/${taskId}/status`, { status: newStatus });
    await selectProject(currentProject.id);
}

async function assignTask(taskId, userId) {
    const payload = { assignedToUserId: userId ? parseInt(userId) : null };
    await client.patch(`/projects/${currentProject.id}/tasks/${taskId}/assign`, payload);
    await selectProject(currentProject.id);
}

async function deleteProject() {
    if(confirm("¿Eliminar proyecto y todo su contenido?")) {
        await client.delete(`/projects/${currentProject.id}`);
        currentProject = null;
        document.getElementById('project-detail').classList.add('hidden');
        document.getElementById('empty-state').classList.remove('hidden');
        await loadProjectsList();
    }
}

async function deleteTask(taskId) {
    if(confirm("¿Eliminar esta tarea?")) {
        await client.delete(`/projects/${currentProject.id}/tasks/${taskId}`);
        await selectProject(currentProject.id);
    }
}

// CONTROL DE MODALES
function openModal(title, fieldsHtml, onSubmit) {
    document.getElementById('modal-title').textContent = title;
    document.getElementById('modal-fields').innerHTML = fieldsHtml;
    const modal = document.getElementById('generic-modal');
    modal.classList.remove('hidden');

    document.getElementById('modal-form').onsubmit = async (e) => {
        e.preventDefault();
        await onSubmit();
        closeModal();
    };
}

function closeModal() {
    document.getElementById('generic-modal').classList.add('hidden');
    document.getElementById('modal-submit-btn').classList.remove('hidden');
}

function openCreateProjectModal() {
    const html = `
        <div class="form-group">
            <label>Nombre</label>
            <input type="text" id="m-proj-name" data-test="modal-input-project-name" required>
        </div>
        <div class="form-group">
            <label>Descripción</label>
            <textarea id="m-proj-desc" data-test="modal-input-project-desc"></textarea>
        </div>
    `;
    openModal("Nuevo Proyecto", html, async () => {
        const name = document.getElementById('m-proj-name').value;
        const description = document.getElementById('m-proj-desc').value;
        await client.post('/projects', { name, description });
        await loadProjectsList();
    });
}

function openEditProjectModal() {
    const html = `
        <div class="form-group">
            <label>Nombre</label>
            <input type="text" id="m-proj-name" value="${currentProject.name}" data-test="modal-input-project-name" required>
        </div>
        <div class="form-group">
            <label>Descripción</label>
            <textarea id="m-proj-desc" data-test="modal-input-project-desc">${currentProject.description || ''}</textarea>
        </div>
    `;
    openModal("Editar Proyecto", html, async () => {
        const name = document.getElementById('m-proj-name').value;
        const description = document.getElementById('m-proj-desc').value;
        await client.put(`/projects/${currentProject.id}`, { name, description });
        await selectProject(currentProject.id);
        await loadProjectsList();
    });
}

function openCreateTaskModal() {
    const html = `
        <div class="form-group">
            <label>Título</label>
            <input type="text" id="m-task-title" data-test="modal-input-task-title" required>
        </div>
        <div class="form-group">
            <label>Descripción</label>
            <textarea id="m-task-desc" data-test="modal-input-task-desc"></textarea>
        </div>
    `;
    openModal("Nueva Tarea", html, async () => {
        const title = document.getElementById('m-task-title').value;
        const description = document.getElementById('m-task-desc').value;
        await client.post(`/projects/${currentProject.id}/tasks`, { title, description });
        await selectProject(currentProject.id);
    });
}

function openEditTaskModal(id, currentTitle, currentDesc) {
    const html = `
        <div class="form-group">
            <label>Título</label>
            <input type="text" id="m-task-title" value="${currentTitle}" data-test="modal-input-task-title" required>
        </div>
        <div class="form-group">
            <label>Descripción</label>
            <textarea id="m-task-desc" data-test="modal-input-task-desc">${currentDesc}</textarea>
        </div>
    `;
    openModal("Editar Tarea", html, async () => {
        const title = document.getElementById('m-task-title').value;
        const description = document.getElementById('m-task-desc').value;
        await client.put(`/projects/${currentProject.id}/tasks/${id}`, { title, description });
        await selectProject(currentProject.id);
    });
}

async function openProfileModal() {
    // Refrescamos los datos por si algo cambió desde el inicio de la sesión
    try {
        const profile = await client.get('/auth/me');

        const createdAt = profile.createdAt
            ? new Date(profile.createdAt).toLocaleString('es-CO', {
                dateStyle: 'long',
                timeStyle: 'short'
            })
            : 'No disponible';

        const html = `
            <ul class="profile-info-list" data-test="profile-info-list">
                <li>
                    <span class="label">ID</span>
                    <span class="value" data-test="profile-id">${profile.id}</span>
                </li>
                <li>
                    <span class="label">Usuario</span>
                    <span class="value" data-test="profile-username">${profile.username}</span>
                </li>
                <li>
                    <span class="label">Correo electrónico</span>
                    <span class="value" data-test="profile-email">${profile.email}</span>
                </li>
                <li>
                    <span class="label">Miembro desde</span>
                    <span class="value" data-test="profile-created-at">${createdAt}</span>
                </li>
            </ul>
        `;

        // El modal genérico envuelve los campos en un <form> con botón "Guardar".
        // Para perfil de solo lectura ocultamos ese botón.
        document.getElementById('modal-submit-btn').classList.add('hidden');

        openModal("Mi Perfil", html, async () => { /* no-op, solo lectura */ });
    } catch (err) {
        alert('No se pudo cargar el perfil: ' + err.message);
    }
}